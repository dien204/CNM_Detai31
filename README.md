# User Trust Platform
## Identity Trust, Risk and Behavior Analytics

Ứng dụng demo đánh giá độ tin cậy người dùng dựa trên dữ liệu giao dịch, hành vi đăng nhập, thay đổi thiết bị/IP, feedback, audit log và monitoring. Project có giao diện Streamlit, backend FastAPI, database SQLite, mô hình ML, giải thích mô hình và một số thành phần MLOps cơ bản.

---

## 1. Tài khoản demo

```text
username: admin
password: Admin@123
```

Người dùng có thể đăng ký tài khoản mới hoặc đặt lại mật khẩu ở màn hình đăng nhập.

---

## 2. Chạy local bằng Docker

Mở terminal tại thư mục project:

```bash
cd CNM-user-trust-score-main
```

Build và chạy:

```bash
docker compose up --build
```

Mở app:

```text
http://localhost:8501
```

Backend API:

```text
http://localhost:8000
```

Chạy ở chế độ nền:

```bash
docker compose up --build -d
```

Xem log:

```bash
docker compose logs -f
```

Dừng project:

```bash
docker compose down
```

Build lại sạch:

```bash
docker compose down --rmi local
docker compose build --no-cache
docker compose up
```

---

## 3. Chạy test bằng Docker

Khi container đang chạy:

```bash
docker compose exec backend pytest -q
```

Kết quả mong muốn:

```text
13 passed
```

---

## 4. Reset database demo

```bash
docker compose exec backend python scripts/init_demo_db.py --reset --n-users 100 --n-transactions 5000
```

Sau đó reload lại:

```text
http://localhost:8501
```

Database SQLite được lưu tại:

```text
data/app/user_trust.db
```

Khi chạy Docker, thư mục `data/app` được mount ra ngoài container nên dữ liệu tài khoản, audit log và feedback vẫn được giữ sau khi tắt/mở lại.

---

## 5. Chức năng chính

### 5.1. Tổng quan

Trang tổng quan hiển thị:

- Tổng giao dịch demo
- Trust Score trung bình
- Điểm hành vi người dùng trung bình trên 100
- Phân bố High / Medium / Low Trust
- User/giao dịch cần ưu tiên xem
- Biểu đồ phân bố rủi ro
- Biểu đồ Trust Score theo dòng

### 5.2. Nhập dữ liệu

Tất cả upload CSV được gom tại trang **Nhập dữ liệu**.

Trang này hỗ trợ:

- Upload CSV giao dịch
- Mapping cột
- Kiểm tra dữ liệu
- Hiển thị số dòng, số cột map được, số cột bỏ qua, trạng thái hợp lệ
- Lưu dữ liệu upload vào session
- Lưu vào database nếu cần

Sau khi upload, các trang phân tích sẽ ưu tiên dùng dữ liệu CSV hiện tại.

### 5.3. Dự đoán và phân tích

Trang này gộp chức năng **Dự đoán và phân tích** và **Giải thích mô hình** vào cùng một nơi để tránh trùng lặp.

Nội dung gồm:

- Chọn dòng giao dịch trong CSV đã upload
- Fraud Probability
- Trust Score
- Risk Level
- Thông tin giao dịch đầu vào
- Giải thích mô hình bằng SHAP nếu khả dụng
- Fallback explanation bằng feature importance nếu SHAP không khả dụng
- Audit log dự đoán

### 5.4. Xem theo từng user

Trang này gộp:

- Hồ sơ người dùng
- Phân tích hành vi người dùng

Nội dung gồm:

- Tổng dòng dataset
- User đang chọn
- Số giao dịch của user
- Trust Score trung bình
- Giao dịch rủi ro
- Thiết bị/IP
- Đăng nhập
- Giao dịch gần đây
- Hành vi theo thời gian

Nếu chưa upload CSV, hệ thống dùng dữ liệu demo để tránh hiển thị số liệu rỗng.

### 5.5. Case study

Mô phỏng các kịch bản:

- User ổn định
- User rủi ro cao
- User nghi ngờ bị chiếm tài khoản

Từ Case study có thể mở hồ sơ user hoặc gửi user sang Feedback. Khi đi từ Case study sang Hồ sơ hoặc Feedback, cuối trang sẽ có nút quay lại Case study.

### 5.6. Feedback

Trang Feedback dùng để review thủ công:

- Chọn user
- Chọn giao dịch liên quan
- Chọn quyết định review
- Nhập ghi chú
- Lưu feedback vào database
- Ghi audit log
- User thường chỉ xem feedback do chính tài khoản đó tạo
- Admin có thể xem toàn bộ feedback trên dữ liệu của user

### 5.7. Monitoring & Drift

Trang này theo dõi:

- Prediction count
- Trust Score trung bình
- Risk distribution
- Drift report
- Model version

### 5.8. Audit log

Audit log dành cho Admin để theo dõi/truy vết thao tác hệ thống. User thường không thấy mục Audit log trong menu.

Ghi nhận các thao tác:

- Login/logout
- Dự đoán
- Import dữ liệu
- Feedback
- Reset database
- Thao tác quản trị

### 5.9. Quản trị hệ thống

Dành cho Admin:

- Xem các bảng trong SQLite
- Xem ma trận quyền
- Reset database demo
- Ghi audit kiểm tra hệ thống

---

## 6. Luồng xử lý chính

### 6.1. Luồng upload và dự đoán

```text
Upload CSV
   ↓
Mapping + validation
   ↓
Chuẩn hóa dữ liệu
   ↓
Model XGBoost
   ↓
Fraud Probability
   ↓
Trust Score
   ↓
Risk Level
   ↓
Hiển thị kết quả + Audit log
```

### 6.2. Luồng xem theo user

```text
Dataset upload hoặc demo
   ↓
Chọn user
   ↓
Hồ sơ user
   ↓
Hành vi user
   ↓
Giao dịch rủi ro
   ↓
Biểu đồ và bảng chi tiết
```

### 6.3. Luồng feedback

```text
Case study / User review
   ↓
Chọn user + giao dịch
   ↓
Nhập quyết định review
   ↓
Lưu feedback
   ↓
Ghi audit log
```

---

## 7. MLOps cơ bản

Project có các thành phần MLOps cơ bản:

- Dockerfile
- Docker Compose
- GitHub Actions CI
- Pytest
- Monitoring & Drift
- Model Card
- Audit log
- Feedback loop
- SHAP/fallback explainability

File CI nằm tại:

```text
.github/workflows/ci.yml
```

CI sẽ tự chạy khi push code lên GitHub:

```text
checkout code → cài Python → cài requirements → chạy pytest
```

---

## 8. Sửa lỗi và tối ưu hiện tại

Bản hiện tại đã xử lý thêm:

- Docker frontend/backend gửi JSON an toàn hơn.
- Tự chuyển `NaN`, `inf`, `-inf` thành `null` khi gọi API.
- Upload CSV mới sẽ xóa cache scoring cũ để tránh dùng nhầm kết quả.
- Chuyển trang có vòng tròn loading nhỏ thay vì để bóng mờ trang cũ quá lâu.
- Trang chủ có thêm thẻ điểm hành vi người dùng trung bình trên 100.
- Trang phân tích/xem user luôn hiện tổng dòng dataset, không để số liệu về 0 khi có dữ liệu demo.

---

## 9. Cấu trúc thư mục chính

```text
CNM-user-trust-score-main/
├── app/
│   ├── streamlit_app.py
│   ├── ui_helpers.py
│   ├── ui_components.py
│   └── static/styles.css
├── backend/main.py
├── src/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── database.py
│   ├── behavior.py
│   ├── explainability.py
│   ├── drift_monitoring.py
│   └── data_validation.py
├── data/
│   ├── demo/
│   └── app/user_trust.db
├── models/
├── docs/
├── scripts/
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## 10. Lệnh thường dùng

Chạy app:

```bash
docker compose up --build
```

Chạy nền:

```bash
docker compose up --build -d
```

Dừng:

```bash
docker compose down
```

Test:

```bash
docker compose exec backend pytest -q
```

Reset DB:

```bash
docker compose exec backend python scripts/init_demo_db.py --reset --n-users 100 --n-transactions 5000
```

---

## 11. Hạn chế

- Dữ liệu chủ yếu là demo/synthetic.
- SQLite phù hợp demo, chưa phù hợp production lớn.
- MLOps mới ở mức cơ bản.
- Chưa có MLflow/DVC/model registry thật.
- Chưa có auto retraining.
- Chưa triển khai cloud production.

---

## 12. Hướng phát triển

- Thay SQLite bằng PostgreSQL.
- Thêm MLflow để quản lý model version.
- Thêm DVC để quản lý dataset.
- Tự động retrain khi drift vượt ngưỡng.
- Triển khai cloud.
- Bổ sung dashboard realtime.

---

## 13. Kết luận

Project đủ để demo một hệ thống dự đoán độ tin cậy và phân tích hành vi người dùng ở mức đồ án cuối kỳ. Hệ thống có frontend, backend, database, model ML, feedback, audit log, monitoring, explainability, Docker và CI cơ bản.

## Cập nhật giao diện và luồng điều hướng

- Trang chủ không hiển thị ô chọn chức năng phụ trợ để tránh rối giao diện.
- Ô chọn chức năng phụ trợ chỉ hiển thị tại trang **Nhập dữ liệu**, sau khi chọn mục sẽ tự chuyển trang.
- Khi chuyển trang, hệ thống hiển thị vòng tròn loading nhỏ thay vì để lại bóng mờ nội dung trang cũ.
- Trang chủ có thẻ **Hành vi user demo xx/100** hiển thị giống kiểu Trust Score của phần dự đoán giao dịch.
- Trang dự đoán giao dịch hiển thị ngay **Tổng dòng dataset** của CSV đang phân tích.
- Khu vực mapping cột trong trang Nhập dữ liệu được tách khoảng cách rõ hơn để không bị dính vào các thẻ validation phía trên.


## Cập nhật trợ lý dữ liệu

Trợ lý trong giao diện đã được mở rộng để trả lời nhiều nhóm câu hỏi hơn, không chỉ hỏi về CSV cơ bản.

Các nhóm câu hỏi hỗ trợ:

- Tổng quan dataset, số dòng, số cột, số feature.
- CSV upload, mapping cột, feature bị thiếu và validation.
- Trust Score, Fraud Probability và Risk Level.
- User rủi ro nhất trong database hoặc CSV đang phân tích.
- Hỏi chi tiết theo mã user, ví dụ `User U0003 có rủi ro không?`.
- Drift monitoring và các cảnh báo phân phối dữ liệu.
- SHAP/explainability và feature importance.
- Feedback, audit log và case study.
- Docker, CI, pytest và MLOps cơ bản.
- Gợi ý nội dung viết báo cáo, hạn chế và hướng phát triển.

Trợ lý hoạt động theo hướng rule-based có dùng dữ liệu hiện tại của hệ thống, gồm dữ liệu demo, CSV upload nếu có, kết quả scoring, behavior database và drift report. Nếu chưa upload CSV, trợ lý vẫn có thể trả lời dựa trên dữ liệu demo và database SQLite.

## Cập nhật trợ lý dữ liệu

- Trợ lý chỉ xuất hiện sau khi người dùng upload CSV ở trang **Nhập dữ liệu**.
- Trợ lý ưu tiên trả lời theo dữ liệu CSV đang dùng trong phiên hiện tại, không phân tích dữ liệu demo khi chưa có upload.
- Thẻ trợ lý nổi bật ở góc dưới phải và có thể mở/đóng khi cần.
- Các nhóm câu hỏi hỗ trợ: số dòng CSV, mapping/validation, Trust Score, user/giao dịch rủi ro, SHAP, Docker, CI, MLOps, báo cáo, hạn chế và hướng phát triển.

## Cập nhật tối ưu tốc độ chuyển trang

Bản hiện tại preload dữ liệu sau khi đăng nhập: model, dữ liệu demo, dashboard sample, monitoring, audit và drift report được làm nóng cache ngay từ đầu. Lần vào đầu tiên có thể chờ lâu hơn một chút, nhưng các lần chuyển trang sau sẽ nhanh hơn và ổn định hơn.

Các phần nặng như SHAP vẫn không nên tính toàn bộ ngay từ đầu; khi deploy online nên đặt `ENABLE_SHAP=0` để giảm thời gian tải.


### Cập nhật trợ lý

Trợ lý chỉ xuất hiện sau khi người dùng upload CSV. Trợ lý tập trung trả lời các câu hỏi liên quan đến dataset đang upload và hệ thống: số dòng/cột, cột thiếu, Trust Score, user rủi ro, drift, feedback, audit log và cách hệ thống hoạt động. Giao diện chat đã bỏ các câu hỏi mẫu để người dùng tự nhập câu hỏi trực tiếp.
