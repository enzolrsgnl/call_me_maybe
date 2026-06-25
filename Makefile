install:
	uv sync

run:
	uv run python3 -m src

debug:
	uv run python3 -m pdb -m src

clean:
	rm -r -f __pycache__ .mypy_cache

lint:
	uv run python3 -m flake8 .
	uv run python3 -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run python3 -m flake8 .
	uv run python3 -m mypy . --strict