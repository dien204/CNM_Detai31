from src.database import (
    authenticate_app_account,
    create_app_account,
    get_connection,
    reset_account_password,
    seed_demo_database,
)


def test_registered_user_persists_and_can_reset_password(tmp_path):
    db_path = str(tmp_path / "auth.db")

    ok, message, account = create_app_account("Nguyen Van A", "vana@example.com", "oldpass123", db_path=db_path)
    assert ok, message
    assert account["role"] == "User"

    ok, user = authenticate_app_account("vana@example.com", "oldpass123", db_path=db_path)
    assert ok
    assert user["username"] == "vana@example.com"

    ok, message = reset_account_password("vana@example.com", "newpass123", db_path=db_path)
    assert ok, message

    ok, _ = authenticate_app_account("vana@example.com", "oldpass123", db_path=db_path)
    assert not ok
    ok, user = authenticate_app_account("vana@example.com", "newpass123", db_path=db_path)
    assert ok
    assert user["role"] == "User"

    # Demo reset should not delete manually registered login accounts.
    seed_demo_database(db_path=db_path, reset=True, n_users=10, n_transactions=100)
    ok, user = authenticate_app_account("vana@example.com", "newpass123", db_path=db_path)
    assert ok
    assert user["role"] == "User"

    conn = get_connection(db_path)
    row = conn.execute("SELECT COUNT(*) AS n FROM app_accounts WHERE lower(email)='vana@example.com'").fetchone()
    conn.close()
    assert int(row["n"]) == 1
