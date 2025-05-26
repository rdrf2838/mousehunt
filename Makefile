code-fix:
	uv run isort .
	uv run basedpyright .
	uv run ruff format .
	uv run ruff check --fix .