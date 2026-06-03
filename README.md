# LatenMOT

Pipeline MOT nhẹ:

`Lightweight Detection -> Motion Prediction -> Two-stage Association -> Lost Track Re-activation -> Conditional ReID -> Stable ID`

Checkpoint detector đã có trong thư mục:

```text
best_crossval_v11_non_early_stop (1).pt
```

## Chạy trên Kaggle hoặc Colab

```python
!pip install -q -r requirements.txt
```

```python
!python latenmot_tracker.py \
  --weights "best_crossval_v11_non_early_stop (1).pt" \
  --source "/kaggle/input/your-video/video.mp4" \
  --output "/kaggle/working/latenmot_output.mp4" \
  --save-mot "/kaggle/working/latenmot_tracks.txt" \
  --device 0 \
  --imgsz 640 \
  --person-class 0
```

Trên Colab, thay `--source` bằng đường dẫn video trong `/content/...` và `--output` bằng `/content/latenmot_output.mp4`.

## Ý tưởng chính

- YOLO/Ultralytics load trực tiếp trọng số `.pt` để detect người.
- Kalman filter dự đoán chuyển động khi object bị che/mất detection ngắn hạn.
- Association stage 1 nối track đang active với detection confidence cao.
- Association stage 2 dùng detection confidence thấp hơn để cứu track đang active.
- Lost track được giữ trong `--track-buffer` frame, thử re-activate trước khi cấp ID mới.
- Conditional ReID dùng color-hist appearance embedding nhẹ, chỉ kích hoạt khi cần nối lost track hoặc cập nhật gallery cho track đã match.
