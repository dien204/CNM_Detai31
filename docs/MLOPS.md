# MLOps tối thiểu cho đồ án User Trust Score

## 1. Pipeline

```text
Raw CSV/Kaggle
   -> src.preprocess
      - split train/validation/test trước khi fit preprocessing
      - fit missing rule, category mapping, median trên train only
      - xuất processed_train.csv, processed_val.csv, processed_test.csv
   -> src.train
      - train XGBoost
      - chọn threshold trên validation
      - đánh giá cuối trên test
      - lưu model, feature_columns, training_metrics
   -> backend/main.py + app/streamlit_app.py
      - inference API/UI
      - audit log, feedback, behavior scoring
   -> monitoring
      - drift report từ dữ liệu mới/database
      - risk distribution và audit trail
```

## 2. Guardrails đã có

- Tách validation riêng để tránh chọn threshold trên test set.
- Preprocessing fit trên train only để giảm data leakage.
- Feature alignment khi inference: thiếu cột thì fill theo metadata, thừa cột thì bỏ qua.
- Unknown category được mã hóa thành `-1`, giúp batch upload không bị crash.
- Audit log và feedback loop hỗ trợ truy vết quyết định.
- CI GitHub Actions chạy pytest sau mỗi push/pull request.

## 3. Những điểm nên nói rõ trong báo cáo

- Dữ liệu demo/hành vi là mô phỏng để trình diễn hệ thống end-to-end.
- SQLite phù hợp demo local; production nên dùng PostgreSQL/MySQL.
- Drift monitoring hiện là thống kê nhẹ, chưa thay thế hệ monitoring production như Evidently/WhyLabs.
- SHAP được bật mặc định trong app bằng `ENABLE_SHAP=1`; nếu môi trường thiếu thư viện hoặc tính toán lỗi, hệ thống tự fallback sang feature importance để demo không bị hỏng. Trong CI có thể đặt `ENABLE_SHAP=0` để test nhanh hơn.

## 4. Hướng nâng cấp production

- Thêm MLflow Model Registry hoặc DVC để versioning dữ liệu/model.
- Thêm scheduled retraining khi drift cao hoặc performance giảm.
- Thêm authentication thật cho API, rate limit và secret management.
- Thêm contract test cho schema dữ liệu từ từng nguồn import.
