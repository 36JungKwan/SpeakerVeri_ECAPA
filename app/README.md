# Demo App - Vietnamese Speaker Recognition

## Chức năng

- Đăng ký người nói mới (nạp nhiều audio cho 1 speaker)
- So sánh 2 audio (cosine similarity + accept/reject)
- Nhận diện 1 audio trong danh sách speaker đã đăng ký
- Hỗ trợ cả tải file audio và ghi âm trực tiếp trên giao diện

## Model đang dùng

- Checkpoint mặc định:
  - `train/outputs/experiments/Mode3_concat_train_raw_mfbe_pitch/best_model.pth`
- Kiến trúc:
  - `mode=3`, `fusion=concat`, `feature_mode=mfbe_pitch`

## Pipeline trích đặc trưng (dựng lại từ notebook)

- Sample rate: 16kHz
- FBank: log-mel (`n_mels=80`, `n_fft=400`, `hop=160`, `center=False`) + CMVN
- Handcrafted: `MFBE + Pitch`
  - MFBE: log-mel + CMVN
  - Pitch: `librosa.pyin` (`fmin=60`, `fmax=500`) + align chiều thời gian + CMVN sau concat

## Chạy app

Từ thư mục root project:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Dữ liệu lưu speaker đã đăng ký

- Mặc định tại: `app/data/speaker_registry.pt`
- Có thể đổi path trong sidebar của app.
