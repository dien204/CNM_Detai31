# Model Card - User Trust / Fraud Risk XGBoost

## Mục tiêu

Model dự đoán xác suất giao dịch có rủi ro gian lận. Trust Score được tính bằng:

```text
Trust Score = (1 - fraud_probability) * 100
```

Sau đó hệ thống kết hợp thêm hành vi đăng nhập, thiết bị/IP, giao dịch, feedback và audit log để đánh giá độ tin cậy dài hạn của người dùng.

## Dữ liệu

- Dataset gốc dự kiến: IEEE-CIS Fraud Detection.
- Dữ liệu demo trong repo dùng để trình diễn UI/API và không đại diện hoàn toàn cho dữ liệu thực tế.
- Với dữ liệu thật, cần đặt file raw vào `data/raw/` rồi chạy `python -m src.preprocess`.

## Chia tập dữ liệu

- Train: dùng để fit preprocessing và train model.
- Validation: dùng để chọn threshold tối ưu theo F1-score.
- Test: chỉ dùng để đánh giá cuối cùng.

Cách chia này giúp tránh data leakage và tránh tối ưu trực tiếp trên test set.

## Metrics chính

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion matrix

Vì bài toán mất cân bằng lớp, PR-AUC, recall và F1 thường quan trọng hơn accuracy.

## Rủi ro và giới hạn

- Model không nên được dùng tự động để khóa tài khoản thật nếu chưa có kiểm duyệt người dùng/pháp lý.
- Dữ liệu thiếu hoặc sai mapping có thể làm score lệch.
- Drift dữ liệu làm performance giảm theo thời gian.
- Cần quy trình feedback/retraining khi triển khai thực tế.
