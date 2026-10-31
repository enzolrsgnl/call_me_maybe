.PHONY: install run debug clean lint lint-strict

install:
	pip install -r requirements.txt

run:
	python3 -m flyin $(MAP)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	python3 -m flake8 .
	python3 -m mypy --explicit-package-bases src

lint-strict:
	python3 -m flake8 .
	python3 -m mypy --strict --explicit-package-bases src

debug:
	python3 -m pdb -m flyin $(MAP)