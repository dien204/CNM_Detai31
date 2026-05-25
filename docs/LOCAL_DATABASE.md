# Local database policy

Project này vẫn sử dụng SQLite ở đường dẫn mặc định:

```text
data/app/user_trust.db
```

File này dùng để lưu dữ liệu runtime của bản demo/local, gồm tài khoản đăng nhập, hồ sơ người dùng, lịch sử dự đoán, feedback, audit log và model registry.

## Vì sao vẫn giữ file database trong bản nộp này?

Bản nộp này giữ lại `data/app/user_trust.db` để người chấm có thể mở project và thấy dữ liệu demo/tài khoản demo ngay, không làm thay đổi cấu trúc gốc của project.

## Tài khoản demo

```text
username: admin
password: Admin@123
```

Các tài khoản đăng ký mới trong giao diện sẽ tiếp tục được lưu vào cùng database này.

## Khi chạy local

Nếu muốn dùng database mặc định:

```bash
python scripts/init_demo_db.py
```

Nếu muốn tạo lại dữ liệu demo từ đầu:

```bash
python scripts/init_demo_db.py --reset
```

Nếu muốn đổi nơi lưu database mà không sửa code:

```bash
export TRUST_DB_PATH=data/app/user_trust.db
```

Trên Windows PowerShell:

```powershell
$env:TRUST_DB_PATH="data/app/user_trust.db"
```

## Khi đưa lên GitHub hoặc production

Không nên commit database thật có dữ liệu người dùng cá nhân. Với môi trường thật, nên dùng PostgreSQL/MySQL và cấu hình bằng biến môi trường. SQLite phù hợp cho demo và đồ án local.
