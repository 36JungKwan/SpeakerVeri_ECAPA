"""
Speaker Verification Model with Hybrid Dual-Stream Architecture
Main Branch: FBank -> ECAPA-TDNN
Auxiliary Branch: Handcrafted (MFBE/Pitch) -> Conv1D
Fusion: Late Fusion (Embedding level 512-dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Bỏ import PTM cũ, thêm các constant từ config mới
from config import (
    FBANK_DIM, HANDCRAFTED_DIM,
    ECAPA_CHANNELS, ECAPA_BLOCKS, ECAPA_KERNEL_SIZE, ECAPA_DILATION,
    EMBEDDING_DIM, AAM_MARGIN, AAM_SCALE,
    MODE, FUSION_METHOD, DIM_MAP
)

# ============================================================================
# ECAPA-TDNN BACKBONE (Nhánh 1 - Nhận FBank)
# ============================================================================
class BottleneckBlock(nn.Module):
    """Bottleneck block for ECAPA-TDNN"""
    def __init__(self, channels=ECAPA_CHANNELS, kernel_size=ECAPA_KERNEL_SIZE, dilation=ECAPA_DILATION):
        super().__init__()
        self.conv1x1_1 = nn.Conv1d(channels, 128, kernel_size=1)
        self.conv1d = nn.Conv1d(
            128, 128, kernel_size=kernel_size, padding=(kernel_size - 1) * dilation // 2, dilation=dilation
        )
        self.conv1x1_2 = nn.Conv1d(128, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1x1_1(x))
        x = self.relu(self.conv1d(x))
        x = self.conv1x1_2(x)
        x = x + residual
        return self.bn(x)


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN xử lý độ dài chuỗi (T) và trả về vector Embedding 1D"""
    def __init__(self, input_dim=FBANK_DIM, channels=ECAPA_CHANNELS, blocks=ECAPA_BLOCKS, kernel_size=ECAPA_KERNEL_SIZE, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.inst_norm = nn.InstanceNorm1d(input_dim) 
        
        self.conv1d_1 = nn.Conv1d(input_dim, channels, kernel_size=5, padding=2)
        self.bn_1 = nn.BatchNorm1d(channels)

        self.blocks = nn.ModuleList([
            BottleneckBlock(channels, kernel_size, dilation=d)
            for d in [1, 2, 3, 4][:blocks]
        ])

        self.conv1d_last = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.fc1 = nn.Linear(channels * 4, embedding_dim)
        self.bn_fc = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        """x: (Batch, FBank_Dim, Time) -> Output: (Batch, Embedding_Dim)"""
        x = self.inst_norm(x)
        x = self.bn_1(self.conv1d_1(x))
        
        for block in self.blocks:
            x = block(x)

        x = self.conv1d_last(x)

        # Attentive / Statistical Pooling
        x_f32 = x.float()
        mean = x_f32.mean(dim=-1)
        std = x_f32.std(dim=-1)
        pooled = torch.cat([mean, std], dim=1).type_as(x)  # (B, channels*4)

        # Xuất vector (Late Fusion level)
        return self.bn_fc(self.fc1(pooled))


# ============================================================================
# HANDCRAFTED ENCODER (Nhánh 2 - Nhận MFBE/Pitch)
# ============================================================================
class HandcraftedEncoder(nn.Module):
    """
    Xử lý tín hiệu thanh điệu/MFBE qua Conv1D nhẹ, sau đó Pooling thành vector 1D
    để ghép cặp ở giai đoạn Late Fusion.
    """
    def __init__(self, input_dim=HANDCRAFTED_DIM, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        # Nối mean+std (512*2) và ép về Embedding_dim
        self.fc = nn.Linear(512 * 2, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.conv_blocks(x)
        
        # BẢO VỆ TRÀN SỐ
        x_f32 = x.float()
        mean = x_f32.mean(dim=-1)
        std = x_f32.std(dim=-1)
        pooled = torch.cat([mean, std], dim=1).type_as(x)
        
        return self.bn(self.fc(pooled))


# ============================================================================
# LATE FUSION MODULES (Ghép nối 2 vector 512-dim)
# ============================================================================
class ConcatenationFusion(nn.Module):
    def __init__(self, dim=EMBEDDING_DIM):
        super().__init__()
        # Ghép 512 + 512 = 1024, sau đó FC về 512
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, emb1, emb2):
        combined = torch.cat([emb1, emb2], dim=1)
        return self.fc(combined)


class GatingFusion(nn.Module):
    def __init__(self, dim=EMBEDDING_DIM):
        super().__init__()
        # Mạng học trọng số Gating từ cả 2 nhánh
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, emb_fbank, emb_hc):
        combined = torch.cat([emb_fbank, emb_hc], dim=1)
        gate_weights = self.gate(combined) # Tensor (B, 512) chứa giá trị (0 -> 1)
        
        # Soft-mix: w * FBank + (1-w) * Handcrafted
        fused = gate_weights * emb_fbank + (1.0 - gate_weights) * emb_hc
        return fused, gate_weights


class CrossAttentionFusion(nn.Module):
    """
    Vector-level Cross Attention. 
    Dùng FBank làm Query để soi chiếu và trích xuất thông tin bổ sung từ Handcrafted (Key/Value).
    """
    def __init__(self, dim=EMBEDDING_DIM):
        super().__init__()
        # Tạo các lớp chiếu (Projection) độc lập cho Q, K, V
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, emb_fbank, emb_hc):
        # 1. emb_fbank đóng vai trò Query (Người soi chiếu)
        q = self.W_q(emb_fbank)  # (Batch, 512)
        
        # 2. emb_hc đóng vai trò Key và Value (Nguồn thông tin bổ sung)
        k = self.W_k(emb_hc)     # (Batch, 512)
        v = self.W_v(emb_hc)     # (Batch, 512)
        
        # 3. Tính điểm tương quan (Attention Score) giữa FBank và Handcrafted
        # Phép nhân vô hướng (Dot product) tạo ra độ lớn tương quan, sau đó chia scale
        scores = torch.sum(q * k, dim=-1, keepdim=True) / math.sqrt(q.size(-1)) # (Batch, 1)
        
        # 4. Tính Attention Weight
        # Dùng Sigmoid thay cho Softmax vì ta chỉ có 1 phần tử (quyết định "lấy nhiều" hay "lấy ít")
        attn_weights = torch.sigmoid(scores) # (Batch, 1)
        
        # 5. Trích xuất Value dựa trên trọng số Attention
        attn_out = attn_weights * v # (Batch, 512)
        
        # 6. Residual Connection: Giữ lại nền tảng FBank, cộng thêm phần thông tin chú ý từ Pitch/MFBE
        out = self.norm(emb_fbank + attn_out)
        return out


# ============================================================================
# COMPLETE MODEL (Wrapper)
# ============================================================================
class SpeakerVerificationModel(nn.Module):
    def __init__(self, num_speakers, mode=MODE, fusion_method=FUSION_METHOD, feature_mode="mfbe_pitch"):
        super().__init__()
        self.mode = mode
        self.fusion_method = fusion_method
        actual_hc_dim = DIM_MAP.get(feature_mode, 81)

        # Mode 1: Chỉ chạy ECAPA-TDNN trên FBank
        if mode in [1, 3]:
            self.main_encoder = ECAPATDNN(input_dim=FBANK_DIM, embedding_dim=EMBEDDING_DIM)

        # Mode 2: Chỉ chạy Conv1D trên Handcrafted
        if mode in [2, 3]:
            self.aux_encoder = HandcraftedEncoder(input_dim=actual_hc_dim, embedding_dim=EMBEDDING_DIM)

        # Mode 3: Gọi thêm hàm Fusion
        if mode == 3:
            if fusion_method == "concat":
                self.fusion = ConcatenationFusion(dim=EMBEDDING_DIM)
            elif fusion_method == "gating":
                self.fusion = GatingFusion(dim=EMBEDDING_DIM)
            elif fusion_method == "cross_attention":
                self.fusion = CrossAttentionFusion(dim=EMBEDDING_DIM)
            else:
                raise ValueError(f"Fusion method không hợp lệ: {fusion_method}")

    def forward(self, return_gates=False, **kwargs):
        gate_weights = None

        if self.mode == 1:
            fbank = kwargs["fbank"]
            speaker_embedding = self.main_encoder(fbank)

        elif self.mode == 2:
            hc_feat = kwargs["handcrafted"]
            speaker_embedding = self.aux_encoder(hc_feat)

        elif self.mode == 3:
            fbank = kwargs["fbank"]
            hc_feat = kwargs["handcrafted"]
            
            emb_fbank = self.main_encoder(fbank)
            emb_hc = self.aux_encoder(hc_feat)
            
            # Giai đoạn Late Fusion
            if self.fusion_method == "gating":
                speaker_embedding, gate_weights = self.fusion(emb_fbank, emb_hc)
            else:
                speaker_embedding = self.fusion(emb_fbank, emb_hc)

        if return_gates and gate_weights is not None:
            return None, speaker_embedding, gate_weights
        else:
            return None, speaker_embedding


# ============================================================================
# AAM-SOFTMAX LOSS (Giữ nguyên gốc vì đã chuẩn)
# ============================================================================
class AAMSoftmaxLoss(nn.Module):
    def __init__(self, num_speakers, embedding_dim=EMBEDDING_DIM, margin=AAM_MARGIN, scale=AAM_SCALE):
        super(AAMSoftmaxLoss, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_speakers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.scale = scale

    def forward(self, logits, labels, embeddings=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-7, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        loss = F.cross_entropy(output, labels)
        
        return loss, output


# ============================================================================
# UTILITIES
# ============================================================================
def get_model(num_speakers, device="cuda", mode=MODE, fusion_method=FUSION_METHOD, feature_mode="mfbe_pitch"):
    # Truyền thêm feature_mode vào class
    model = SpeakerVerificationModel(num_speakers, mode=mode, fusion_method=fusion_method, feature_mode=feature_mode)
    model = model.to(device)

    print(f"\n{'='*70}")
    print(f"Khởi tạo mô hình thành công (Hybrid ECAPA-TDNN)")
    print(f"  Mode: {mode} (1=Main, 2=Handcrafted, 3=Fusion)")
    if mode == 3:
        print(f"  Fusion method (Late Fusion): {fusion_method.upper()}")
    # Thêm dòng log này để dễ theo dõi feature đang dùng
    if mode in [2,3]:
        print(f"  Handcrafted Feature: {feature_mode.upper()}")
    print(f"  Num speakers: {num_speakers}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*70}\n")
    return model