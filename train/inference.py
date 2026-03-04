import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm

try:
    from .metrics import compute_eer, compute_mindcf
    from .dataset import DualStreamDataset, collate_fn_dual
    from .model import get_model
    from .config import DEVICE, MODE, FUSION_METHOD, FEATURE_MODE, BATCH_SIZE
except ImportError:
    from metrics import compute_eer, compute_mindcf
    from dataset import DualStreamDataset, collate_fn_dual
    from model import get_model
    from config import DEVICE, MODE, FUSION_METHOD, FEATURE_MODE, BATCH_SIZE

def evaluate_speaker_verification(model, data_loader, device, num_pairs=20000, p_target=0.05):
    model.eval()
    all_embeddings = []
    all_labels = []

    print("\n[Evaluation] Extracting embeddings...")
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Extracting", leave=False):
            labels = batch_data["label"]
            inputs = {k: v.to(device) for k, v in batch_data.items() if k != "label"}
            
            _, embeddings = model(**inputs) 
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    all_embeddings = F.normalize(torch.cat(all_embeddings, dim=0), p=2, dim=1)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    print(f"[Evaluation] Generating {num_pairs} balanced pairs for scoring...")
    
    # Nhóm các file âm thanh theo từng người nói (Speaker ID)
    label_to_indices = defaultdict(list)
    for idx, lbl in enumerate(all_labels):
        label_to_indices[lbl].append(idx)
        
    unique_labels = list(label_to_indices.keys())

    valid_pos_labels = [lbl for lbl in unique_labels if len(label_to_indices[lbl]) >= 2]
    if len(valid_pos_labels) == 0:
        raise ValueError("Inference requires at least one speaker with >=2 samples to create positive pairs.")
    if len(unique_labels) < 2:
        raise ValueError("Inference requires at least 2 distinct speakers to create negative pairs.")

    max_pos_pairs = sum(len(v) * (len(v) - 1) // 2 for v in label_to_indices.values())
    max_neg_pairs = (len(all_labels) * (len(all_labels) - 1) // 2) - max_pos_pairs
    if max_neg_pairs <= 0:
        raise ValueError("Inference cannot build negative pairs from this dataset.")

    requested_pos = num_pairs // 2
    requested_neg = num_pairs - requested_pos
    num_pos = min(requested_pos, max_pos_pairs)
    num_neg = min(requested_neg, max_neg_pairs)
    if num_pos + num_neg <= 0:
        raise ValueError("Inference cannot build any evaluation pairs from this dataset.")
    
    idx1_list, idx2_list, y_true = [], [], []
    
    # 1. TẠO POSITIVE PAIRS (50% - Cùng 1 người)
    pos_count = 0
    while pos_count < num_pos:
        lbl = random.choice(valid_pos_labels)
        indices = label_to_indices[lbl]
        if len(indices) >= 2:
            i1, i2 = random.sample(indices, 2)
            idx1_list.append(i1)
            idx2_list.append(i2)
            y_true.append(1)
            pos_count += 1
            
    # 2. TẠO NEGATIVE PAIRS (50% - Khác người)
    neg_count = 0
    while neg_count < num_neg:
        lbl1, lbl2 = random.sample(unique_labels, 2)
        i1 = random.choice(label_to_indices[lbl1])
        i2 = random.choice(label_to_indices[lbl2])
        idx1_list.append(i1)
        idx2_list.append(i2)
        y_true.append(0)
        neg_count += 1
        
    # Tính Cosine Similarity một lần cho tất cả các cặp đã chọn
    emb1 = all_embeddings[idx1_list]
    emb2 = all_embeddings[idx2_list]
    
    scores = torch.sum(emb1 * emb2, dim=1).numpy()
    y_true = np.array(y_true)
    
    eer, eer_thresh = compute_eer(y_true, scores)
    min_dcf, dcf_thresh = compute_mindcf(y_true, scores, p_target=p_target)
    
    return {
        "Num Pairs": int(num_pos + num_neg),
        "EER (%)": float(eer * 100),
        "EER Threshold": float(eer_thresh),
        f"MinDCF (p={p_target})": float(min_dcf),
        "MinDCF Threshold": float(dcf_thresh)
    }


def run_inference(args):
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = DualStreamDataset(args.base_dir, mode=args.mode, feature_mode=args.feature_mode)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn_dual, mode=args.mode, is_train=False),
        pin_memory=(device.type == "cuda"),
    )

    num_speakers = dataset.num_speakers
    model = get_model(
        num_speakers=num_speakers,
        device=str(device),
        mode=args.mode,
        fusion_method=args.fusion_method,
        feature_mode=args.feature_mode,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_speaker_verification(
        model=model,
        data_loader=data_loader,
        device=device,
        num_pairs=args.num_pairs,
        p_target=args.p_target,
    )

    print("\n[Inference] Done")
    for key, value in metrics.items():
        print(f"- {key}: {value}")

    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate speaker verification checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to feature shards directory")
    parser.add_argument("--mode", type=int, default=MODE, choices=[1, 2, 3])
    parser.add_argument("--fusion_method", type=str, default=FUSION_METHOD, choices=["concat", "gating", "cross_attention"])
    parser.add_argument("--feature_mode", type=str, default=FEATURE_MODE, choices=["mfbe_pitch", "mfbe_only", "pitch_only"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_pairs", type=int, default=20000)
    parser.add_argument("--p_target", type=float, default=0.05)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    arguments = parser.parse_args()
    run_inference(arguments)