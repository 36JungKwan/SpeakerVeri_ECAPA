"""
Speaker Verification Model with Hybrid Dual-Stream Architecture
Main Branch: FBank -> ECAPA-TDNN
Auxiliary Branch: Handcrafted (MFBE/Pitch) -> Conv1D
Fusion: Late Fusion (Embedding level 512-dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math

# Bỏ import PTM cũ, thêm các constant từ config mới
try:
    from .config import (
        FBANK_DIM, HANDCRAFTED_DIM,
        ECAPA_CHANNELS, ECAPA_BLOCKS, ECAPA_KERNEL_SIZE, ECAPA_DILATION,
        EMBEDDING_DIM, AAM_MARGIN, AAM_SCALE,
        MODE, FUSION_METHOD, DIM_MAP
    )
except ImportError:
    from config import (
        FBANK_DIM, HANDCRAFTED_DIM,
        ECAPA_CHANNELS, ECAPA_BLOCKS, ECAPA_KERNEL_SIZE, ECAPA_DILATION,
        EMBEDDING_DIM, AAM_MARGIN, AAM_SCALE,
        MODE, FUSION_METHOD, DIM_MAP
    )

# ============================================================================
# ECAPA-TDNN BACKBONE (Nhánh 1 - Nhận FBank)
# ============================================================================
class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class ECAPATDNN_Backbone(nn.Module):
    def __init__(self, channels=ECAPA_CHANNELS, embedding_dim=EMBEDDING_DIM):
        super(ECAPATDNN_Backbone, self).__init__()
        self.specaug = FbankAug()
        
        # Conv đầu vào nhận 80-dim FBank
        self.conv1 = nn.Conv1d(80, channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        
        self.layer1 = Bottle2neck(channels, channels, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(channels, channels, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(channels, channels, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * channels, 1536, kernel_size=1)
        
        # Attentive Statistics Pooling
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, embedding_dim) 
        self.bn6 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        # Tự động bật/tắt Augmentation dựa vào mode model.train() hay model.eval()
        if self.training:
            x = self.specaug(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        
        x = F.relu(self.layer4(torch.cat((x1, x2, x3), dim=1)))
        t = x.size()[-1]
        
        global_x = torch.cat((x, x.mean(dim=2, keepdim=True).repeat(1, 1, t), 
                             torch.sqrt(x.var(dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))
        
        emb = self.bn6(self.fc6(self.bn5(torch.cat((mu, sg), 1))))
        return emb


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
            self.main_encoder = ECAPATDNN_Backbone(channels=ECAPA_CHANNELS, embedding_dim=EMBEDDING_DIM)

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
        self.scale = scale
        self.update_margin(margin)

    def update_margin(self, new_margin):
        """Hàm này sẽ được gọi từ train.py mỗi epoch để tăng dần margin"""
        self.margin = new_margin
        self.cos_m = math.cos(new_margin)
        self.sin_m = math.sin(new_margin)
        self.th = math.cos(math.pi - new_margin)
        self.mm = math.sin(math.pi - new_margin) * new_margin

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