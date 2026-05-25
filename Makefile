.PHONY: install test run-api run-app init-db preprocess train evaluate clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

test:
	pytest -q

run-api:
	python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

run-app:
	python -m streamlit run app/streamlit_app.py

init-db:
	python scripts/init_demo_db.py --reset --n-users 100 --n-transactions 5000

preprocess:
	python -m src.preprocess

train:
	python -m src.train

evaluate:
	python -m src.evaluate

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -f data/app/*.db data/app/*.db-*
