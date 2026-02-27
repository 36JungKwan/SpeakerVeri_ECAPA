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
    MODE, FUSION_METHOD
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
        self.conv1d_1 = nn.Conv1d(input_dim, channels, kernel_size=5, padding=2)
        self.bn_1 = nn.BatchNorm1d(channels)

        self.blocks = nn.ModuleList([
            BottleneckBlock(channels, kernel_size, dilation=d)
            for d in [1, 2, 3, 4][:blocks] # Tăng dần dilation
        ])

        self.conv1d_last = nn.Conv1d(channels, channels * 2, kernel_size=1)
        # Sinh ra vector 1D bằng FC layer cuối cùng
        self.fc1 = nn.Linear(channels * 4, embedding_dim)  # *4 vì nối mean+std
        self.bn_fc = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        """x: (Batch, FBank_Dim, Time) -> Output: (Batch, Embedding_Dim)"""
        x = self.bn_1(self.conv1d_1(x))
        
        for block in self.blocks:
            x = block(x)

        x = self.conv1d_last(x)

        # Attentive / Statistical Pooling
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)
        x = torch.cat([mean, std], dim=1)  # (B, channels*4)

        # Xuất vector (Late Fusion level)
        x = self.bn_fc(self.fc1(x))
        return x


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
        """x: (Batch, HC_Dim, Time) -> Output: (Batch, Embedding_Dim)"""
        x = self.conv_blocks(x)
        
        # Pooling tương tự nhánh chính
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)
        x = torch.cat([mean, std], dim=1)
        
        x = self.bn(self.fc(x))
        return x


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
    def __init__(self, dim=EMBEDDING_DIM, num_heads=8):
        super().__init__()
        # Lấy FBank làm Query, Handcrafted làm Key/Value
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, emb_fbank, emb_hc):
        # Biến vector (B, D) thành sequence length 1: (B, 1, D)
        q = emb_fbank.unsqueeze(1)
        k = emb_hc.unsqueeze(1)
        v = emb_hc.unsqueeze(1)

        attn_out, _ = self.mha(query=q, key=k, value=v)
        
        # Residual Connection (Giữ lại gốc FBank cộng thêm thông tin chú ý từ Pitch)
        out = self.norm(emb_fbank + attn_out.squeeze(1))
        return out


# ============================================================================
# COMPLETE MODEL (Wrapper)
# ============================================================================
class SpeakerVerificationModel(nn.Module):
    def __init__(self, num_speakers, mode=MODE, fusion_method=FUSION_METHOD):
        super().__init__()
        self.mode = mode
        self.fusion_method = fusion_method

        # Mode 1: Chỉ chạy ECAPA-TDNN trên FBank
        if mode in [1, 3]:
            self.main_encoder = ECAPATDNN(input_dim=FBANK_DIM, embedding_dim=EMBEDDING_DIM)

        # Mode 2: Chỉ chạy Conv1D trên Handcrafted
        if mode in [2, 3]:
            self.aux_encoder = HandcraftedEncoder(input_dim=HANDCRAFTED_DIM, embedding_dim=EMBEDDING_DIM)

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
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
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
def get_model(num_speakers, device="cuda", mode=MODE, fusion_method=FUSION_METHOD):
    model = SpeakerVerificationModel(num_speakers, mode=mode, fusion_method=fusion_method)
    model = model.to(device)

    print(f"\n{'='*70}")
    print(f"Khởi tạo mô hình thành công (Hybrid ECAPA-TDNN)")
    print(f"  Mode: {mode} (1=Main, 2=Handcrafted, 3=Fusion)")
    if mode == 3:
        print(f"  Fusion method (Late Fusion): {fusion_method.upper()}")
    print(f"  Num speakers: {num_speakers}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*70}\n")
    return model