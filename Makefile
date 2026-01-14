# Makefile for Federated Stress Prediction project
# Usage: make <target>
# Example: make test, make lint, make format

.PHONY: help install install-dev install-all test lint format type-check clean run-baseline run-federated notebooks pre-commit docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  make install         - Install core dependencies"
	@echo "  make install-dev     - Install development dependencies"
	@echo "  make install-all     - Install all dependencies (dev + jupyter + monitoring)"
	@echo "  make test            - Run tests with pytest"
	@echo "  make test-cov        - Run tests with coverage report"
	@echo "  make lint            - Run linting (ruff)"
	@echo "  make format          - Format code with black"
	@echo "  make type-check      - Run type checking with mypy"
	@echo "  make check-all       - Run all checks (format, lint, type, test)"
	@echo "  make clean           - Remove generated files"
	@echo "  make clean-data      - Remove processed data (WARNING: irreversible)"
	@echo "  make run-baseline    - Train baseline model"
	@echo "  make run-federated   - Train federated model"
	@echo "  make notebooks       - Start Jupyter Lab"
	@echo "  make pre-commit      - Run pre-commit hooks on all files"
	@echo "  make docs            - Generate documentation"

# Installation targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[dev,jupyter,monitoring,demo]"

# Testing targets
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow and not gpu"

test-gpu:
	pytest tests/ -v -m "gpu"

# Code quality targets
lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

format:
	black src/ tests/ scripts/

format-check:
	black --check src/ tests/ scripts/

type-check:
	mypy src/ --ignore-missing-imports

security-check:
	bandit -r src/ -c pyproject.toml

# Combined checks
check-all: format-check lint type-check test

# Cleaning targets
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -f .coverage

clean-data:
	@echo "⚠️  WARNING: This will delete all processed data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/processed/*; \
		rm -rf data/interim/*; \
		echo "✅ Processed data removed"; \
	else \
		echo "❌ Cancelled"; \
	fi

clean-models:
	@echo "⚠️  WARNING: This will delete all model checkpoints!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf checkpoints/*; \
		rm -rf outputs/*; \
		echo "✅ Model checkpoints removed"; \
	else \
		echo "❌ Cancelled"; \
	fi

# Training targets
run-baseline:
	python src/training/train_baseline.py --config configs/baseline_config.yaml

run-federated:
	python src/training/train_federated.py --config configs/federated_config.yaml

run-personalized:
	python src/training/train_personalized.py --config configs/personalization_config.yaml

# Data processing
preprocess:
	python scripts/preprocess_data.py --config configs/baseline_config.yaml

download-data:
	bash scripts/download_wesad.sh

# Development tools
notebooks:
	jupyter lab --no-browser --port=8888

demo:
	streamlit run demo/streamlit_app.py

# Pre-commit
pre-commit:
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate

# Documentation
docs:
	@echo "Generating documentation..."
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Docker targets (if you decide to use Docker later)
docker-build:
	docker build -t federated-stress-prediction .

docker-run:
	docker run -it --rm --gpus all -v $$(pwd)/data:/app/data federated-stress-prediction

# Development workflow shortcuts
dev-setup: install-dev pre-commit-install
	@echo "✅ Development environment ready!"

pre-commit-install:
	pre-commit install

# Quick experiment runner
experiment:
	@read -p "Experiment name: " exp_name; \
	python src/training/train_baseline.py \
		--config configs/baseline_config.yaml \
		--experiment-name "$$exp_name"

# Monitoring
tensorboard:
	tensorboard --logdir=runs --port=6006

wandb-sync:
	wandb sync runs/

# Git shortcuts
git-status:
	git status

git-log:
	git log --oneline --graph --all -20

# Environment info
env-info:
	@echo "Python version:"
	@python --version
	@echo ""
	@echo "PyTorch version:"
	@python -c "import torch; print(torch.__version__)"
	@echo ""
	@echo "CUDA available:"
	@python -c "import torch; print(torch.cuda.is_available())"
	@echo ""
	@echo "Device count:"
	@python -c "import torch; print(torch.cuda.device_count())"

# Create project structure
init-structure:
	mkdir -p src/data src/models src/federated src/training src/evaluation src/utils
	mkdir -p tests notebooks configs scripts demo docs
	mkdir -p data/raw data/processed data/fitbit
	mkdir -p checkpoints outputs logs results
	touch src/__init__.py src/data/__init__.py src/models/__init__.py
	touch src/federated/__init__.py src/training/__init__.py src/evaluation/__init__.py
	@echo "✅ Project structure created"
