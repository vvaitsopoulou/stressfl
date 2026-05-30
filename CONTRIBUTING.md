# Contributing to Federated Stress Prediction

Thank you for your interest in this project! While this is an individual dissertation project, feedback, suggestions, and contributions are welcome.

## 📝 Note on Individual Project

This is a Bachelor's dissertation project by Vasiliki Anna Vaitsopoulou. The primary development will be done by the author, but:

- **Bug reports** are very welcome
- **Feature suggestions** that align with the project scope are appreciated
- **Code contributions** can be submitted but may not be merged immediately due to academic integrity requirements
- **Documentation improvements** are always helpful

## 🐛 Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in the [Issues](https://github.com/vvaitsopoulou/stressfl/issues) tab
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Your environment (Python version, OS, etc.)
   - Relevant code snippets or error messages

## 💡 Suggesting Features

For feature requests:

1. Check existing issues to avoid duplicates
2. Explain the use case and why it would be valuable
3. Consider whether it aligns with the dissertation scope

## 🔧 Development Setup

If you want to contribute code or experiment with the project:

### 1. Fork and Clone

```bash
git clone https://github.com/vvaitsopoulou/stressfl.git
cd stressfl
```

### 2. Set up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development tools
pip install -e ".[dev]"

# Optional: Jupyter, monitoring, demo
pip install -e ".[jupyter,monitoring,demo]"
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

### 5. Set up Environment Variables

```bash
cp .env.example .env
# Edit .env with your configurations
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run only fast tests (skip slow/GPU tests)
pytest -m "not slow and not gpu"
```

## 🎨 Code Style

This project uses several tools to maintain code quality:

- **Black** for code formatting (line length: 88)
- **Ruff** for linting (replaces flake8, isort, etc.)
- **MyPy** for type checking
- **Bandit** for security checks

### Format Code

```bash
# Format all Python files
black src/ tests/

# Check formatting without changing files
black --check src/ tests/
```

### Lint Code

```bash
# Lint with Ruff
ruff check src/ tests/

# Auto-fix issues where possible
ruff check --fix src/ tests/
```

### Type Check

```bash
mypy src/
```

### Run All Checks

```bash
# Pre-commit will run all checks
pre-commit run --all-files
```

## 📋 Coding Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use Black's default settings (line length 88)
- Add type hints where reasonable
- Write docstrings for functions and classes (Google style)

### Example Function

```python
def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: str = "cuda",
) -> dict[str, float]:
    """Train a PyTorch model.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader containing training data.
        epochs: Number of training epochs.
        device: Device to train on ('cuda' or 'cpu').

    Returns:
        Dictionary containing training metrics (loss, accuracy, etc.).

    Raises:
        ValueError: If epochs is less than 1.
    """
    if epochs < 1:
        raise ValueError("Epochs must be at least 1")

    # Training logic here
    metrics = {"loss": 0.0, "accuracy": 0.0}
    return metrics
```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(federated): add FedProx aggregation algorithm
fix(data): handle missing values in WESAD preprocessing
docs(readme): update installation instructions
test(models): add unit tests for baseline model
```

## 🔀 Pull Request Process

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding guidelines

3. **Write tests** for new functionality

4. **Ensure all checks pass:**
   ```bash
   pre-commit run --all-files
   pytest
   ```

5. **Update documentation** if needed

6. **Commit your changes** with clear commit messages

7. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** with:
   - Clear description of changes
   - Link to related issues
   - Screenshots/examples if applicable

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages are clear and follow conventions
- [ ] No merge conflicts with main branch

## 📚 Project Structure

When adding new code, follow the existing structure:

- `src/data/`: Data loading and preprocessing
- `src/models/`: Model architectures
- `src/federated/`: Federated learning logic
- `src/training/`: Training scripts
- `src/evaluation/`: Metrics and visualization
- `tests/`: Unit and integration tests
- `notebooks/`: Exploratory Jupyter notebooks
- `configs/`: YAML configuration files

## ❓ Questions?

Feel free to:
- Open an issue for questions
- Reach out via email: vaisanna97@gmail.com
- Check the [Discussions](https://github.com/vvaitsopoulou/stressfl/discussions) tab

## 🙏 Acknowledgments

Thank you for helping improve this project! Every contribution, no matter how small, is valued.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
