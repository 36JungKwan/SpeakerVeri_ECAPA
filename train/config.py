"""
Configuration file for Speaker Verification Training - Hybrid ECAPA-TDNN
"""
import os

# ============================================================================
# DATA PATHS (Cập nhật theo cấu trúc mới)
# ============================================================================
# Đường dẫn tới thư mục gốc chứa dữ liệu
TRAIN_VAL_DIR = "path/to/data/train_val"
TEST_DIR = "path/to/data/test"

# Tên các thư mục con chứa feature (phải khớp chính xác tên thư mục trên ổ cứng)
FBANK_FOLDER = "fbank_shards"
HANDCRAFTED_FOLDERS = {
    "mfbe_pitch": "mfbe_pitch_shards",
    "mfbe_only": "mfbe_shards",
    "pitch_only": "pitch_shards",
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Mode: 1 (FBank + ECAPA-TDNN), 2 (Handcrafted + Conv1D), 3 (Hybrid 2 nhánh)
MODE = 3

# Phương pháp Fusion cho MODE 3: "concat" hoặc "attention"
FUSION_METHOD = "concat"

# Chọn loại Handcrafted feature sẽ sử dụng
FEATURE_MODE = "mfbe_pitch"  # Keys từ dictionary HANDCRAFTED_FOLDERS ở trên

# Định nghĩa số chiều (Channels) của từng loại feature
FBANK_DIM = 80  # Đầu vào chuẩn cho ECAPA-TDNN
DIM_MAP = {
    "mfbe_pitch": 81,  # 80 MFBE + 1 Pitch
    "mfbe_only": 80,
    "pitch_only": 1,
    "mfcc_only": 40
}
# HANDCRAFTED_DIM = DIM_MAP.get(FEATURE_MODE, 81)

# Kiến trúc nhánh ECAPA-TDNN
ECAPA_CHANNELS = 512
ECAPA_BLOCKS = 4
ECAPA_KERNEL_SIZE = 5
ECAPA_DILATION = 1

# Kích thước Speaker Embedding cuối cùng (Trước lớp Classification)
EMBEDDING_DIM = 512

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
MIN_LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.0001

# Early stopping
EARLY_STOP_PATIENCE = 10
EARLY_STOP_DELTA = 1e-4

# Learning rate scheduler
LR_SCHEDULER = "plateau"  # "cosine" hoặc "plateau"
COSINE_T_MAX = 50
PLATEAU_PATIENCE = 5
PLATEAU_FACTOR = 0.5

# ============================================================================
# AAM-SOFTMAX LOSS
# ============================================================================
AAM_MARGIN = 0.2  # m margin
AAM_SCALE = 30    # s scale

# ============================================================================
# OPTIMIZATION & DEVICE
# ============================================================================
OPTIMIZER = "adam"
MOMENTUM = 0.9
NESTEROV = True
DEVICE = "cuda"
MIXED_PRECISION = True

# ============================================================================
# LOGGING & SPLITTING
# ============================================================================
CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_NAME = "best_model.pth"
FINAL_MODEL_NAME = "final_model.pth"
TRAIN_RATIO = 0.85
RANDOM_SEED = 42