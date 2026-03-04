"""
Training script for Dual-Stream Speaker Verification
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import numpy as np
import os
import json
import shutil
from datetime import datetime
from tqdm import tqdm
from torchinfo import summary
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

try:
    from .inference import evaluate_speaker_verification
    from .config import (
        TRAIN_VAL_DIR,
        BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MIN_LEARNING_RATE, WEIGHT_DECAY,
        EARLY_STOP_PATIENCE, EARLY_STOP_DELTA, LR_SCHEDULER, COSINE_T_MAX,
        PLATEAU_PATIENCE, PLATEAU_FACTOR, OPTIMIZER, MOMENTUM, NESTEROV,
        DEVICE, MIXED_PRECISION, CHECKPOINT_DIR,
        BEST_MODEL_NAME, FINAL_MODEL_NAME, MODE, FUSION_METHOD, FEATURE_MODE,
        FBANK_DIM, EMBEDDING_DIM, DIM_MAP,
    )
    from .model import get_model, AAMSoftmaxLoss
    from .dataset import create_train_val_loaders
except ImportError:
    from inference import evaluate_speaker_verification
    from config import (
        TRAIN_VAL_DIR,
        BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MIN_LEARNING_RATE, WEIGHT_DECAY,
        EARLY_STOP_PATIENCE, EARLY_STOP_DELTA, LR_SCHEDULER, COSINE_T_MAX,
        PLATEAU_PATIENCE, PLATEAU_FACTOR, OPTIMIZER, MOMENTUM, NESTEROV,
        DEVICE, MIXED_PRECISION, CHECKPOINT_DIR,
        BEST_MODEL_NAME, FINAL_MODEL_NAME, MODE, FUSION_METHOD, FEATURE_MODE,
        FBANK_DIM, EMBEDDING_DIM, DIM_MAP,
    )
    from model import get_model, AAMSoftmaxLoss
    from dataset import create_train_val_loaders

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_margin(epoch, final_margin=0.35, increase_epochs=15):
    """
    Tăng dần margin từ 0.0 lên final_margin trong increase_epochs đầu tiên.
    """
    if epoch >= increase_epochs:
        return final_margin
    return (epoch / increase_epochs) * final_margin

def save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, map_location=DEVICE):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"], checkpoint["best_loss"]

def compute_metrics(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    return (predictions == labels).float().mean().item()

class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE, delta=EARLY_STOP_DELTA):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None # Đổi best_loss thành best_score
        self.early_stop = False

    def __call__(self, val_eer): # Truyền val_eer vào
        if self.best_score is None:
            self.best_score = val_eer
        elif val_eer > self.best_score - self.delta: # EER không giảm
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_eer
            self.counter = 0

# ============================================================================
# TRAINING & VALIDATION LOOPS
# ============================================================================
def train_epoch(model, train_loader, optimizer, criterion, scaler, epoch, device, use_mixed_precision=False):
    model.train()
    total_loss, total_accuracy, num_batches = 0.0, 0.0, 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)

    for batch_data in progress_bar:
        labels = batch_data["label"].to(device)
        inputs = {k: v.to(device) for k, v in batch_data.items() if isinstance(v, torch.Tensor) and k != "label"}

        if "fbank" in inputs and torch.isnan(inputs["fbank"]).any(): continue
        if "handcrafted" in inputs and torch.isnan(inputs["handcrafted"]).any(): continue

        optimizer.zero_grad()
        
        if use_mixed_precision:
            with autocast(device_type=device.type, enabled=True):
                _, embeddings = model(**inputs)
                loss, logits = criterion(None, labels, embeddings=embeddings.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, embeddings = model(**inputs)
            loss, logits = criterion(None, labels, embeddings=embeddings.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        accuracy = compute_metrics(logits, labels)
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

        if num_batches % 10 == 0:
            progress_bar.set_postfix({"loss": f"{total_loss/num_batches:.4f}", "acc": f"{total_accuracy/num_batches:.4f}"})

    return total_loss / max(1, num_batches), total_accuracy / max(1, num_batches)

# ============================================================================
# GATING ANALYSIS
# ============================================================================
def analyze_gating_behavior(model, loader, device, exp_dir):
    if model.mode != 3 or model.fusion_method != "gating":
        return None, None
    
    model.eval()
    all_gates, all_labels = [], []
    
    print("\nAnalyzing gating weights...")
    with torch.no_grad():
        for batch_data in tqdm(loader, leave=False):
            labels = batch_data["label"].to(device)
            inputs = {k: v.to(device) for k, v in batch_data.items() if k != "label"}
            
            _, _, gate_weights = model(return_gates=True, **inputs)
            
            # Với Late Fusion, gate_weights là vector (Batch, 512). Lấy trung bình trọng số của từng sample.
            gate_avg = gate_weights.mean(dim=-1).cpu().numpy()
            all_gates.extend(gate_avg)
            all_labels.extend(labels.cpu().numpy())
    
    all_gates = np.array(all_gates)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_gates, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', label='Neutral (0.5)')
    ax.set_xlabel('Gate Value (FBank Weight)')
    ax.set_ylabel('Frequency')
    ax.set_title('Gating Behavior: FBank vs Handcrafted\n(>0.5: Trust FBank, <0.5: Trust Handcrafted)')
    ax.legend()
    
    os.makedirs(os.path.join(exp_dir, "gating_analysis"), exist_ok=True)
    plt.savefig(os.path.join(exp_dir, "gating_analysis", "gate_distribution.png"))
    plt.close()
    
    return all_gates, all_labels

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    use_mixed_precision = bool(args.mixed_precision and device.type == "cuda")

    if args.mixed_precision and device.type != "cuda":
        print("⚠ mixed_precision=True nhưng device không phải CUDA, tự động tắt mixed precision.")

    if args.exp_name is None:
        args.exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    exp_dir = os.path.join(args.output_dir, "experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, "tensorboard_logs"))

    # Config snapshot
    config_snapshot = vars(args).copy()
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # Dataloaders
    train_loader, val_loader, speaker_to_idx, num_speakers = create_train_val_loaders(
        args.base_dir, args.mode, args.feature_mode, args.batch_size, num_workers=0
    )

    # Model
    model = get_model(num_speakers, device=str(device), mode=args.mode, fusion_method=args.fusion_method, feature_mode=args.feature_mode)

    # Model Summary
    try:
        actual_hc_dim = DIM_MAP.get(args.feature_mode, 81)
        dummy_kwargs = {}
        if args.mode in [1,3]: 
            dummy_kwargs["fbank"] = torch.randn(args.batch_size, FBANK_DIM, 300).to(device)
        if args.mode in [2,3]: 
            dummy_kwargs["handcrafted"] = torch.randn(args.batch_size, actual_hc_dim, 300).to(device)
            
        # Dùng **dummy_kwargs thay vì input_size
        model_summary = summary(model, **dummy_kwargs, verbose=0)
        
        with open(os.path.join(exp_dir, "model_summary.txt"), "w") as f:
            f.write(str(model_summary))
    except Exception as e:
        print(f"⚠ Could not save model summary: {e}")

    # Loss & Optimizer
    criterion = AAMSoftmaxLoss(num_speakers=num_speakers, embedding_dim=EMBEDDING_DIM).to(device)
    params = list(model.parameters()) + list(criterion.parameters())
    
    if args.optimizer.lower() == "adam":
        opt = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        opt = optim.SGD(params, lr=args.learning_rate, momentum=MOMENTUM, nesterov=NESTEROV, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(opt, T_max=COSINE_T_MAX, eta_min=MIN_LEARNING_RATE) if args.lr_scheduler == "cosine" else ReduceLROnPlateau(opt, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE)
    scaler = GradScaler(device="cuda", enabled=use_mixed_precision)
    early_stopping = EarlyStopping(patience=args.early_stop_patience)

    best_eer_path = os.path.join(exp_dir, BEST_MODEL_NAME)
    best_mindcf_path = os.path.join(exp_dir, "best_mindcf_model.pth")
    final_model_path = os.path.join(exp_dir, FINAL_MODEL_NAME)

    best_val_eer = float("inf")
    best_val_mindcf = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_eer": [], "val_mindcf": []}

    print("Starting training (Open-set Validation)...\n")
    val_trials_path = os.path.join(args.base_dir, "val_trials.txt")
    if os.path.exists(val_trials_path):
        print(f"✓ Sử dụng cặp validation có sẵn: {val_trials_path}")
    else:
        val_trials_path = None
        print("⚠ Không tìm thấy val_trials.txt, fallback sang random balanced pairs.")

    for epoch in range(args.num_epochs):

        current_margin = get_margin(epoch, final_margin=0.35, increase_epochs=15)
        criterion.update_margin(current_margin)
        
        # 1. Train 1 epoch
        t_loss, t_acc = train_epoch(model, train_loader, opt, criterion, scaler, epoch, device, use_mixed_precision=use_mixed_precision)

        # 2. VALIDATION ĐỘC LẬP BẰNG EER & MinDCF
        print(f"  -> Đang tính toán EER (Open-set) cho Epoch {epoch+1}...")

        val_metrics = evaluate_speaker_verification(
            model=model,
            data_loader=val_loader,
            device=device,
            p_target=0.05,
            trials_path=val_trials_path,
        )
        v_eer = val_metrics["EER (%)"]
        v_mindcf = val_metrics[f"MinDCF (p=0.05)"]
        
        # 3. SCHEDULER & EARLY STOPPING LÚC NÀY BUỘC PHẢI DÙNG EER
        if args.lr_scheduler == "cosine": scheduler.step()
        else: scheduler.step(v_eer) 

        early_stopping(v_eer) # Theo dõi EER

        # 4. LƯU HISTORY & TENSORBOARD
        history["train_loss"].append(t_loss); history["train_acc"].append(t_acc)
        history["val_eer"].append(v_eer); history["val_mindcf"].append(v_mindcf)

        writer.add_scalar("Loss/Train", t_loss, epoch)
        writer.add_scalar("Accuracy/Train", t_acc, epoch)
        writer.add_scalar("Metrics/Val_OpenSet_EER", v_eer, epoch)
        writer.add_scalar("Metrics/Val_OpenSet_MinDCF", v_mindcf, epoch)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {t_loss:.4f}, Acc: {t_acc:.4f} | Val EER: {v_eer:.2f}% | MinDCF: {v_mindcf:.4f} | LR: {opt.param_groups[0]['lr']:.6f}")

        # 5. LƯU BEST MODEL THEO EER & MinDCF
        if v_eer < best_val_eer:
            best_val_eer = v_eer
            save_checkpoint(model, opt, epoch, best_val_eer, best_eer_path)
            
        if v_mindcf < best_val_mindcf:
            best_val_mindcf = v_mindcf
            save_checkpoint(model, opt, epoch, best_val_mindcf, best_mindcf_path)

        if early_stopping.early_stop:
            print("\n✓ Early stopping triggered do EER không giảm nữa!")
            break

    # Final Tasks
    model, _, _, _ = load_checkpoint(best_eer_path, model, map_location=device)
    save_checkpoint(model, opt, epoch, best_val_eer, final_model_path)
    
    # Gating Analysis
    gates, _ = analyze_gating_behavior(model, val_loader, device, exp_dir)

    final_results = {
        "status": "completed",
        "epochs_trained": epoch + 1,
        "config": vars(args),  # Lưu toàn bộ tham số (batch_size, lr, mode, feature...)
        "best_metrics": {
            "best_val_eer": best_val_eer,
            "best_val_mindcf": best_val_mindcf
        }
    }
    history_path = os.path.join(exp_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    # Ghi đè vào file results.json
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(final_results, f, indent=4) # Chỉnh indent=4 cho file JSON dễ đọc hơn

    writer.close()
    return model, history, exp_dir


def build_parser():
    parser = argparse.ArgumentParser(description="Train dual-stream speaker verification model")
    parser.add_argument("--base_dir", type=str, default=TRAIN_VAL_DIR, help="Path to train/val feature directory")
    parser.add_argument("--output_dir", type=str, default=CHECKPOINT_DIR, help="Directory to store outputs")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name. Auto-generated if empty")

    parser.add_argument("--mode", type=int, default=MODE, choices=[1, 2, 3], help="1=FBank, 2=Handcrafted, 3=Fusion")
    parser.add_argument("--fusion_method", type=str, default=FUSION_METHOD, choices=["concat", "gating", "cross_attention"], help="Fusion method for mode=3")
    parser.add_argument("--feature_mode", type=str, default=FEATURE_MODE, choices=list(DIM_MAP.keys()), help="Handcrafted feature mode")

    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER, choices=["adam", "sgd"])
    parser.add_argument("--lr_scheduler", type=str, default=LR_SCHEDULER, choices=["cosine", "plateau"])
    parser.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE)

    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--mixed_precision", type=lambda x: str(x).lower() in ["1", "true", "yes", "y"], default=MIXED_PRECISION)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)