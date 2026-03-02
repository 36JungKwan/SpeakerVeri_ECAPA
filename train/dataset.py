"""
Dataset loader for Dual-stream Speaker Verification (Optimized for 32GB RAM)
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import random
import os
import glob
import gc
from functools import partial
from config import RANDOM_SEED, TRAIN_RATIO, FBANK_FOLDER, HANDCRAFTED_FOLDERS

def load_shard_data(folder_path):
    """Tải Shards với cơ chế nén Float16 để tiết kiệm 50% RAM"""
    print(f"  -> Tối ưu RAM: Đang nạp data từ {os.path.basename(folder_path)}...")
    shard_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.pt"), recursive=True))
    if not shard_files:
        raise ValueError(f"Không tìm thấy file .pt nào trong {folder_path}")
        
    shards_data = []
    all_speaker_ids = []
    sample_map = [] # Lưu vị trí: idx -> (shard_idx, vị trí_trong_shard)
    
    for shard_idx, f in enumerate(shard_files):
        # Tắt cảnh báo weights_only của PyTorch
        data = torch.load(f, map_location='cpu', weights_only=False)
        
        feat_key = [k for k in data.keys() if k not in ["speaker_ids", "filenames", "model_name"]][0]
        features = data[feat_key]
        
        # Ép kiểu toàn bộ về Float16 (Giảm một nửa dung lượng RAM)
        if isinstance(features, torch.Tensor):
            shards_data.append(features.half())
        else:
            shards_data.append([t.half() for t in features])
            
        spks = data["speaker_ids"]
        all_speaker_ids.extend(spks)
        
        # Lưu lại bản đồ tìm kiếm thay vì xé lẻ tensor
        for i in range(len(spks)):
            sample_map.append((shard_idx, i))
            
        # Kích hoạt dọn rác ngay lập tức
        del data
        del features
        gc.collect()
        
    return shards_data, all_speaker_ids, sample_map

class DualStreamDataset(Dataset):
    def __init__(self, base_dir, mode=3, feature_mode="mfbe_pitch", speaker_to_idx=None):
        self.mode = mode
        self.hc_folder_name = HANDCRAFTED_FOLDERS.get(feature_mode, "mfbe_pitch_shards")
        
        fbank_dir = os.path.join(base_dir, FBANK_FOLDER)
        hc_dir = os.path.join(base_dir, self.hc_folder_name)
        
        self.speaker_ids = []
        
        if mode in [1,3]:
            self.fbank_shards, self.speaker_ids, self.fbank_map = load_shard_data(fbank_dir)
            
        if mode in [2,3]:
            self.hc_shards, hc_speaker_ids, self.hc_map = load_shard_data(hc_dir)
            if mode == 2:
                self.speaker_ids = hc_speaker_ids
                
        self.speaker_to_idx = speaker_to_idx or {}
        if not self.speaker_to_idx:
            unique_speakers = sorted(set(self.speaker_ids))
            self.speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
            
        self.num_speakers = len(self.speaker_to_idx)

    def __len__(self):
        return len(self.speaker_ids)

    def __getitem__(self, idx):
        spk_id = self.speaker_ids[idx]
        label = self.speaker_to_idx[spk_id]
        data = {"label": label}
        
        if self.mode in [1,3]:
            shard_idx, local_idx = self.fbank_map[idx]
            # Lấy data ra và bung ngược lại thành Float32 cho GPU
            fbank_feat = self.fbank_shards[shard_idx][local_idx].float()
            if fbank_feat.dim() == 1: fbank_feat = fbank_feat.unsqueeze(0)
            data["fbank"] = fbank_feat
            
        if self.mode in [2,3]:
            shard_idx, local_idx = self.hc_map[idx]
            hc_feat = self.hc_shards[shard_idx][local_idx].float()
            if hc_feat.dim() == 1: hc_feat = hc_feat.unsqueeze(0)
            data["handcrafted"] = hc_feat
            
        return data

def collate_fn_dual(batch, mode, is_train=True, max_frames=300):
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    output = {"label": labels}

    def process_features(features):
        safe_features = [f.unsqueeze(0) if f.dim() == 1 else f for f in features]
        processed = []
        if is_train:
            for f in safe_features:
                c, t = f.shape
                if t > max_frames:
                    start = random.randint(0, t - max_frames)
                    f = f[:, start:start + max_frames]
                elif t < max_frames:
                    f = F.pad(f.unsqueeze(0), (0, max_frames - t), mode='replicate').squeeze(0)
                processed.append(f)
        else:
            max_t = min(max([f.shape[-1] for f in safe_features]), 1000)
            for f in safe_features:
                if f.shape[-1] > max_t: f = f[:, :max_t]
                pad_len = max_t - f.shape[-1]
                if pad_len > 0: f = F.pad(f.unsqueeze(0), (0, pad_len), mode='replicate').squeeze(0)
                processed.append(f)
        return torch.stack(processed)

    if mode in [1,3]: output["fbank"] = process_features([item["fbank"] for item in batch])
    if mode in [2,3]: output["handcrafted"] = process_features([item["handcrafted"] for item in batch])

    return output

def create_train_val_loaders(base_dir, mode=3, feature_mode="mfbe_pitch", batch_size=64, num_workers=0):
    print(f"🔍 Đang quét dữ liệu Train/Val tại: {base_dir}...")
    full_dataset = DualStreamDataset(base_dir, mode, feature_mode)
    num_samples = len(full_dataset)
    speaker_to_idx = full_dataset.speaker_to_idx
    
    indices = list(range(num_samples))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    train_end = int(num_samples * TRAIN_RATIO)
    
    train_loader = DataLoader(
        Subset(full_dataset, indices[:train_end]), batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=partial(collate_fn_dual, mode=mode, is_train=True), pin_memory=True
    )
    
    val_loader = DataLoader(
        Subset(full_dataset, indices[train_end:]), batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=partial(collate_fn_dual, mode=mode, is_train=False), pin_memory=True
    )
    
    return train_loader, val_loader, speaker_to_idx, len(speaker_to_idx)