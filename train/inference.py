import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

try:
    from .metrics import compute_eer, compute_mindcf
except ImportError:
    from metrics import compute_eer, compute_mindcf

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
    
    idx1_list, idx2_list, y_true = [], [], []
    num_pos = num_pairs // 2
    num_neg = num_pairs - num_pos
    
    # 1. TẠO POSITIVE PAIRS (50% - Cùng 1 người)
    pos_count = 0
    while pos_count < num_pos:
        lbl = random.choice(unique_labels)
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
        "EER (%)": float(eer * 100),
        "EER Threshold": float(eer_thresh),
        f"MinDCF (p={p_target})": float(min_dcf),
        "MinDCF Threshold": float(dcf_thresh)
    }