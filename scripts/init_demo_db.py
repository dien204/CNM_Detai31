import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import DB_PATH, seed_demo_database


def main():
    parser = argparse.ArgumentParser(description="Initialize local SQLite demo database.")
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--demo-csv", default="data/demo/demo_transactions.csv")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--n-transactions", type=int, default=5000)
    args = parser.parse_args()

    seed_demo_database(
        db_path=args.db_path,
        demo_csv_path=args.demo_csv,
        reset=args.reset,
        n_users=args.n_users,
        n_transactions=args.n_transactions,
    )
    print(f"Database ready: {args.db_path}")
    print(f"Users: {args.n_users} | Transactions: {args.n_transactions}")


if __name__ == "__main__":
    main()
