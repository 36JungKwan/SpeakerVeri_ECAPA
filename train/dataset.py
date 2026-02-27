"""
Dataset loader for Dual-stream Speaker Verification (FBank & Handcrafted)
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import random
import os
import glob
from config import RANDOM_SEED, TRAIN_RATIO, FBANK_FOLDER, HANDCRAFTED_FOLDERS


class DualStreamDataset(Dataset):
    def __init__(self, base_dir, mode=3, feature_mode="mfbe_pitch", speaker_to_idx=None):
        """
        Args:
            base_dir: Thư mục chứa các folder feature ('FBank', 'MFBE + Pitch',...)
            mode: 1 (Chỉ FBank), 2 (Chỉ Handcrafted), 3 (Cả hai)
            feature_mode: 'mfbe_pitch', 'mfbe_only', v.v. để chọn folder Handcrafted
        """
        self.base_dir = base_dir
        self.mode = mode
        self.hc_folder_name = HANDCRAFTED_FOLDERS.get(feature_mode, "MFBE + Pitch")
        
        # Đường dẫn cụ thể tới 2 thư mục
        self.fbank_dir = os.path.join(base_dir, FBANK_FOLDER)
        self.hc_dir = os.path.join(base_dir, self.hc_folder_name)
        
        # Quét lấy danh sách tất cả file .pt (dựa vào FBank làm gốc)
        self.file_list = []
        search_path = os.path.join(self.fbank_dir, "**", "*.pt")
        for file_path in glob.glob(search_path, recursive=True):
            file_name = os.path.basename(file_path)
            self.file_list.append(file_name)
            
        if len(self.file_list) == 0:
            raise ValueError(f"Không tìm thấy file .pt nào trong {self.fbank_dir}")
            
        self.file_list.sort()
        
        # Xây dựng danh sách Speaker (Giả định Tên file: spk001_utt01.pt -> ID: spk001)
        # Bạn có thể sửa logic split('_')[0] nếu format tên file của bạn khác
        self.speaker_ids = [f.split('_')[0] for f in self.file_list]
        
        # Tạo mapping cho Labels
        self.speaker_to_idx = speaker_to_idx or {}
        if not self.speaker_to_idx:
            unique_speakers = sorted(set(self.speaker_ids))
            self.speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
            
        self.num_speakers = len(self.speaker_to_idx)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        spk_id = self.speaker_ids[idx]
        label = self.speaker_to_idx[spk_id]
        
        data = {"label": label}
        
        # 1. Load FBank (Cho nhánh ECAPA-TDNN)
        if self.mode in [1, 3]:
            fbank_path = os.path.join(self.fbank_dir, file_name)
            fbank_feat = torch.load(fbank_path, map_location='cpu').float()
            # Đảm bảo shape (C, T). Nếu data là (T, C), cần transpose: fbank_feat.T
            if fbank_feat.dim() == 1:
                fbank_feat = fbank_feat.unsqueeze(0)
            data["fbank"] = fbank_feat
            
        # 2. Load Handcrafted Feature (MFBE, Pitch...)
        if self.mode in [2, 3]:
            hc_path = os.path.join(self.hc_dir, file_name)
            if not os.path.exists(hc_path):
                raise FileNotFoundError(f"Lỗi ghép cặp: Không tìm thấy {hc_path}")
            hc_feat = torch.load(hc_path, map_location='cpu').float()
            
            if hc_feat.dim() == 1:
                hc_feat = hc_feat.unsqueeze(0)
            data["handcrafted"] = hc_feat
            
        return data


def collate_fn_dual(batch, mode):
    """Dynamic padding cho batch dựa theo số Time-frames lớn nhất"""
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    output = {"label": labels}

    # Gom và pad FBank
    if mode in [1, 3]:
        fbanks = [item["fbank"] for item in batch]
        max_t = max([f.shape[-1] for f in fbanks])
        
        padded_fbanks = []
        for f in fbanks:
            pad_len = max_t - f.shape[-1]
            if pad_len > 0:
                padded_f = F.pad(f.unsqueeze(0), (0, pad_len), mode='replicate').squeeze(0)
            else:
                padded_f = f
            padded_fbanks.append(padded_f)
        output["fbank"] = torch.stack(padded_fbanks)  # (B, FBANK_DIM, T)

    # Gom và pad Handcrafted
    if mode in [2, 3]:
        hcs = [item["handcrafted"] for item in batch]
        max_t = max([f.shape[-1] for f in hcs])
        
        padded_hcs = []
        for f in hcs:
            pad_len = max_t - f.shape[-1]
            if pad_len > 0:
                padded_f = F.pad(f.unsqueeze(0), (0, pad_len), mode='replicate').squeeze(0)
            else:
                padded_f = f
            padded_hcs.append(padded_f)
        output["handcrafted"] = torch.stack(padded_hcs)  # (B, HC_DIM, T)

    return output


def create_train_val_loaders(base_dir, mode=3, feature_mode="mfbe_pitch", batch_size=64, num_workers=0):
    """Tự động đọc folder train/val, quét labels và chia Subset 85-15"""
    print(f"🔍 Đang quét dữ liệu Train/Val tại: {base_dir}...")
    full_dataset = DualStreamDataset(base_dir, mode, feature_mode)
    num_samples = len(full_dataset)
    speaker_to_idx = full_dataset.speaker_to_idx
    
    indices = list(range(num_samples))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    train_end = int(num_samples * TRAIN_RATIO)
    
    train_loader = DataLoader(
        Subset(full_dataset, indices[:train_end]),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=lambda b: collate_fn_dual(b, mode), pin_memory=True
    )
    
    val_loader = DataLoader(
        Subset(full_dataset, indices[train_end:]),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=lambda b: collate_fn_dual(b, mode), pin_memory=True
    )
    
    return train_loader, val_loader, speaker_to_idx, len(speaker_to_idx)


def create_test_loader(test_dir, mode=3, feature_mode="mfbe_pitch", batch_size=64, num_workers=0):
    """Đọc tập test hoàn toàn độc lập"""
    print(f"🔍 Đang quét dữ liệu Test tại: {test_dir}...")
    test_dataset = DualStreamDataset(test_dir, mode, feature_mode)
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=lambda b: collate_fn_dual(b, mode), pin_memory=True
    )
    return test_loader, len(test_dataset.speaker_to_idx)