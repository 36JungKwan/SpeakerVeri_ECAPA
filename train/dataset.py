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
import copy
from collections import OrderedDict

try:
    from .config import RANDOM_SEED, TRAIN_RATIO, FBANK_FOLDER, HANDCRAFTED_FOLDERS
except ImportError:
    from config import RANDOM_SEED, TRAIN_RATIO, FBANK_FOLDER, HANDCRAFTED_FOLDERS

def scan_shard_metadata(folder_path):
    print(f"  -> Quét metadata shard từ {os.path.basename(folder_path)} (không nạp toàn bộ RAM)...")
    shard_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.pt"), recursive=True))
    if not shard_files:
        raise ValueError(f"Không tìm thấy file .pt nào trong {folder_path}")

    shard_feat_keys = []
    all_speaker_ids = []
    all_utt_ids = []
    sample_map = []
    
    for shard_idx, f in enumerate(shard_files):
        data = torch.load(f, map_location='cpu', weights_only=False)

        feat_key = [k for k in data.keys() if k not in ["speaker_ids", "filenames", "model_name"]][0]
        shard_feat_keys.append(feat_key)
        features = data[feat_key]

        spks = data["speaker_ids"]
        all_speaker_ids.extend(spks)

        filenames = data.get("filenames")
        if filenames is not None and len(filenames) == len(spks):
            all_utt_ids.extend([str(name) for name in filenames])
        else:
            shard_name = os.path.basename(f)
            all_utt_ids.extend([f"{shard_name}::{i}" for i in range(len(spks))])
        
        for i in range(len(spks)):
            sample_map.append((shard_idx, i))

        del data
        del features

    gc.collect()
    return shard_files, shard_feat_keys, all_speaker_ids, all_utt_ids, sample_map


def preload_shard_data(folder_path):
    print(f"  -> Tối ưu tốc độ: Nạp shard vào RAM (Float16) từ {os.path.basename(folder_path)}...")
    shard_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.pt"), recursive=True))
    if not shard_files:
        raise ValueError(f"Không tìm thấy file .pt nào trong {folder_path}")

    shards_data = []
    all_speaker_ids = []
    all_utt_ids = []
    sample_map = []

    for shard_idx, shard_path in enumerate(shard_files):
        data = torch.load(shard_path, map_location='cpu', weights_only=False)
        feat_key = [k for k in data.keys() if k not in ["speaker_ids", "filenames", "model_name"]][0]
        features = data[feat_key]

        if isinstance(features, torch.Tensor):
            shards_data.append(features.half())
        else:
            shards_data.append([t.half() for t in features])

        spks = data["speaker_ids"]
        all_speaker_ids.extend(spks)

        filenames = data.get("filenames")
        if filenames is not None and len(filenames) == len(spks):
            all_utt_ids.extend([str(name) for name in filenames])
        else:
            shard_name = os.path.basename(shard_path)
            all_utt_ids.extend([f"{shard_name}::{i}" for i in range(len(spks))])

        for i in range(len(spks)):
            sample_map.append((shard_idx, i))

        del data
        del features

    gc.collect()
    return shards_data, all_speaker_ids, all_utt_ids, sample_map

class DualStreamDataset(Dataset):
    def __init__(
        self,
        base_dir,
        mode=3,
        feature_mode="mfbe_pitch",
        speaker_to_idx=None,
        cache_strategy="preload",
        max_cached_shards=2,
    ):
        self.mode = mode
        self.cache_strategy = cache_strategy
        self.max_cached_shards = max(1, int(max_cached_shards))
        self.hc_folder_name = HANDCRAFTED_FOLDERS.get(feature_mode, "mfbe_pitch_shards")
        
        fbank_dir = os.path.join(base_dir, FBANK_FOLDER)
        hc_dir = os.path.join(base_dir, self.hc_folder_name)
        
        self.speaker_ids = []

        self.fbank_cache = OrderedDict()
        self.hc_cache = OrderedDict()

        if mode in [1,3]:
            if self.cache_strategy == "preload":
                (
                    self.fbank_shards,
                    self.speaker_ids,
                    self.utt_ids,
                    self.fbank_map,
                ) = preload_shard_data(fbank_dir)
            else:
                (
                    self.fbank_shard_files,
                    self.fbank_feat_keys,
                    self.speaker_ids,
                    self.utt_ids,
                    self.fbank_map,
                ) = scan_shard_metadata(fbank_dir)

        if mode in [2,3]:
            if self.cache_strategy == "preload":
                (
                    self.hc_shards,
                    hc_speaker_ids,
                    hc_utt_ids,
                    self.hc_map,
                ) = preload_shard_data(hc_dir)
            else:
                (
                    self.hc_shard_files,
                    self.hc_feat_keys,
                    hc_speaker_ids,
                    hc_utt_ids,
                    self.hc_map,
                ) = scan_shard_metadata(hc_dir)
            if mode == 2:
                self.speaker_ids = hc_speaker_ids
                self.utt_ids = hc_utt_ids

        if not hasattr(self, "utt_ids"):
            self.utt_ids = [str(i) for i in range(len(self.speaker_ids))]
                
        self.speaker_to_idx = speaker_to_idx or {}
        if not self.speaker_to_idx:
            unique_speakers = sorted(set(self.speaker_ids))
            self.speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}

        self.num_speakers = len(self.speaker_to_idx)

    def _load_cached_shard(self, shard_files, feat_keys, cache_dict, shard_idx):
        if shard_idx in cache_dict:
            cache_dict.move_to_end(shard_idx)
            return cache_dict[shard_idx]

        shard_path = shard_files[shard_idx]
        data = torch.load(shard_path, map_location='cpu', weights_only=False)
        feat_key = feat_keys[shard_idx]
        features = data[feat_key]

        if isinstance(features, torch.Tensor):
            features = features.half()
        else:
            features = [t.half() for t in features]

        cache_dict[shard_idx] = features
        cache_dict.move_to_end(shard_idx)

        while len(cache_dict) > self.max_cached_shards:
            cache_dict.popitem(last=False)

        return features

    def _reset_runtime_cache(self):
        self.fbank_cache = OrderedDict()
        self.hc_cache = OrderedDict()

    def __len__(self):
        return len(self.speaker_ids)

    def __getitem__(self, idx):
        spk_id = self.speaker_ids[idx]
        utt_id = self.utt_ids[idx]
        label = self.speaker_to_idx[spk_id]
        data = {"label": label, "utt_id": utt_id}
        
        if self.mode in [1,3]:
            shard_idx, local_idx = self.fbank_map[idx]
            if self.cache_strategy == "preload":
                fbank_features = self.fbank_shards[shard_idx]
            else:
                fbank_features = self._load_cached_shard(self.fbank_shard_files, self.fbank_feat_keys, self.fbank_cache, shard_idx)
            fbank_feat = fbank_features[local_idx].float()
            if fbank_feat.dim() == 1: fbank_feat = fbank_feat.unsqueeze(0)
            data["fbank"] = fbank_feat

        if self.mode in [2,3]:
            shard_idx, local_idx = self.hc_map[idx]
            if self.cache_strategy == "preload":
                hc_features = self.hc_shards[shard_idx]
            else:
                hc_features = self._load_cached_shard(self.hc_shard_files, self.hc_feat_keys, self.hc_cache, shard_idx)
            hc_feat = hc_features[local_idx].float()
            if hc_feat.dim() == 1: hc_feat = hc_feat.unsqueeze(0)
            data["handcrafted"] = hc_feat

        return data

def collate_fn_dual(batch, mode, is_train=True, max_frames=300):
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    output = {"label": labels, "utt_id": [item["utt_id"] for item in batch]}

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
                    f = F.pad(f.unsqueeze(0), (0, max_frames - t), mode='constant', value=0).squeeze(0)
                processed.append(f)
        else:
            max_t = min(max([f.shape[-1] for f in safe_features]), 1000)
            for f in safe_features:
                if f.shape[-1] > max_t: f = f[:, :max_t]
                pad_len = max_t - f.shape[-1]
                if pad_len > 0: 
                    f = F.pad(f.unsqueeze(0), (0, pad_len), mode='constant', value=0).squeeze(0)
                processed.append(f)

        return torch.stack(processed)

    if mode in [1,3]: output["fbank"] = process_features([item["fbank"] for item in batch])
    if mode in [2,3]: output["handcrafted"] = process_features([item["handcrafted"] for item in batch])

    return output

def create_train_val_loaders(
    base_dir,
    mode=3,
    feature_mode="mfbe_pitch",
    batch_size=64,
    num_workers=0,
    cache_strategy="preload",
    max_cached_shards=2,
):
    print(f"🔍 Đang quét dữ liệu Train/Val tại: {base_dir}...")
    full_dataset = DualStreamDataset(
        base_dir,
        mode,
        feature_mode,
        cache_strategy=cache_strategy,
        max_cached_shards=max_cached_shards,
    )
    
    # 1. LẤY DANH SÁCH SPEAKER VÀ TRỘN NGẪU NHIÊN
    unique_speakers = sorted(set(full_dataset.speaker_ids))
    random.seed(RANDOM_SEED)
    random.shuffle(unique_speakers)

    # 2. CHIA TẬP DATA THEO SPEAKER ID (Open-set)
    num_spks = len(unique_speakers)
    train_spk_end = int(num_spks * TRAIN_RATIO)
    
    train_spks = set(unique_speakers[:train_spk_end])
    val_spks = set(unique_speakers[train_spk_end:])

    # 3. LỌC INDEX TƯƠNG ỨNG
    train_indices = [i for i, spk in enumerate(full_dataset.speaker_ids) if spk in train_spks]
    val_indices = [i for i, spk in enumerate(full_dataset.speaker_ids) if spk in val_spks]

    # 4. TẠO MAPPING NHÃN (LABEL 0 -> N) ĐỘC LẬP
    train_speaker_to_idx = {spk: idx for idx, spk in enumerate(sorted(train_spks))}
    val_speaker_to_idx = {spk: idx for idx, spk in enumerate(sorted(val_spks))}

    # 5. TẠO 2 DATASET ĐỘC LẬP ĐỂ KHÔNG ĐỤNG CHẠM LABEL
    train_dataset = copy.copy(full_dataset)
    train_dataset.speaker_to_idx = train_speaker_to_idx
    train_dataset._reset_runtime_cache()

    val_dataset = copy.copy(full_dataset)
    val_dataset.speaker_to_idx = val_speaker_to_idx
    val_dataset._reset_runtime_cache()

    # 6. TẠO DATALOADER
    train_loader = DataLoader(
        Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=partial(collate_fn_dual, mode=mode, is_train=True), pin_memory=True
    )
    
    val_loader = DataLoader(
        Subset(val_dataset, val_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=partial(collate_fn_dual, mode=mode, is_train=False), pin_memory=True
    )
    
    # Trả về số lượng speaker của tập Train (AAM-Softmax chỉ cần biết con số này)
    return train_loader, val_loader, train_speaker_to_idx, len(train_spks)