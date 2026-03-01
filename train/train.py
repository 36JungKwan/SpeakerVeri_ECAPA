"""
Training script for Dual-Stream Speaker Verification
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
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

from config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MIN_LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOP_PATIENCE, EARLY_STOP_DELTA, LR_SCHEDULER, COSINE_T_MAX,
    PLATEAU_PATIENCE, PLATEAU_FACTOR, OPTIMIZER, MOMENTUM, NESTEROV,
    DEVICE, MIXED_PRECISION, CHECKPOINT_DIR,
    BEST_MODEL_NAME, FINAL_MODEL_NAME, MODE, FUSION_METHOD, FEATURE_MODE,
    AAM_MARGIN, AAM_SCALE, FBANK_DIM, EMBEDDING_DIM, DIM_MAP,
)
from model import get_model, AAMSoftmaxLoss
from dataset import create_train_val_loaders

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
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
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ============================================================================
# TRAINING & VALIDATION LOOPS
# ============================================================================
def train_epoch(model, train_loader, optimizer, criterion, scaler, epoch, device):
    model.train()
    total_loss, total_accuracy, num_batches = 0.0, 0.0, 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)

    for batch_data in progress_bar:
        labels = batch_data["label"].to(device)
        inputs = {k: v.to(device) for k, v in batch_data.items() if k != "label"}

        if "fbank" in inputs and torch.isnan(inputs["fbank"]).any(): continue
        if "handcrafted" in inputs and torch.isnan(inputs["handcrafted"]).any(): continue

        optimizer.zero_grad()
        
        if MIXED_PRECISION:
            with autocast():
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

        if num_batches % LOG_INTERVAL == 0:
            progress_bar.set_postfix({"loss": f"{total_loss/num_batches:.4f}", "acc": f"{total_accuracy/num_batches:.4f}"})

    return total_loss / max(1, num_batches), total_accuracy / max(1, num_batches)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_accuracy, num_batches = 0.0, 0.0, 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        for batch_data in progress_bar:
            labels = batch_data["label"].to(device)
            inputs = {k: v.to(device) for k, v in batch_data.items() if k != "label"}

            _, embeddings = model(**inputs)
            loss, logits = criterion(None, labels, embeddings=embeddings.float())

            total_loss += loss.item()
            total_accuracy += compute_metrics(logits, labels)
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{total_loss/num_batches:.4f}"})

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
    model = get_model(num_speakers, device=str(device), mode=args.mode, fusion_method=args.fusion_method)

    # Model Summary
    try:
        actual_hc_dim = DIM_MAP.get(args.feature_mode, 81)
        # Giả định một độ dài thời gian T=300 để in summary
        if args.mode == 3:
            input_data = {"fbank": (args.batch_size, FBANK_DIM, 300), "handcrafted": (args.batch_size, actual_hc_dim, 300)}
        elif args.mode == 1:
            input_data = {"fbank": (args.batch_size, FBANK_DIM, 300)}
        else:
            input_data = {"handcrafted": (args.batch_size, actual_hc_dim, 300)}
            
        model_summary = summary(model, input_size=input_data, verbose=0, device=str(device))
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
    scaler = GradScaler() if args.mixed_precision else None
    early_stopping = EarlyStopping(patience=args.early_stop_patience)

    best_val_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("Starting training...\n")
    for epoch in range(args.num_epochs):
        t_loss, t_acc = train_epoch(model, train_loader, opt, criterion, scaler, epoch, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)

        if args.lr_scheduler == "cosine": scheduler.step()
        else: scheduler.step(v_loss)

        history["train_loss"].append(t_loss); history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss); history["val_acc"].append(v_acc)

        writer.add_scalar("Loss/Train", t_loss, epoch); writer.add_scalar("Loss/Validation", v_loss, epoch)
        writer.add_scalar("Accuracy/Train", t_acc, epoch); writer.add_scalar("Accuracy/Validation", v_acc, epoch)

        print(f"Epoch {epoch+1:3d} | Train Loss: {t_loss:.4f}, Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f}, Acc: {v_acc:.4f} | LR: {opt.param_groups[0]['lr']:.6f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_checkpoint(model, opt, epoch, best_val_loss, os.path.join(exp_dir, BEST_MODEL_NAME))

        early_stopping(v_loss)
        if early_stopping.early_stop:
            print("\n✓ Early stopping triggered!")
            break

    # Final Tasks
    model, _, _, _ = load_checkpoint(os.path.join(exp_dir, BEST_MODEL_NAME), model)
    
    # Gating Analysis
    gates, _ = analyze_gating_behavior(model, val_loader, device, exp_dir)

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump({"best_val_loss": best_val_loss, "epochs": epoch+1}, f, indent=2)

    writer.close()
    return model, history, exp_dir