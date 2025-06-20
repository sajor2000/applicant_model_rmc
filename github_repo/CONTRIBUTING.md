# Contributing to Rush AI Admissions System

Thank you for your interest in contributing to the Rush Medical College AI Admissions System! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/rush-ai-admissions.git
   cd rush-ai-admissions
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/RushUniversity/ai-admissions.git
   ```

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Check if the issue already exists
- Include:
  - Clear description of the bug
  - Steps to reproduce
  - Expected behavior
  - Actual behavior
  - System information

### Suggesting Features

- Open a discussion first
- Provide use cases
- Consider implementation complexity
- Align with project goals

### Code Contributions

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Add tests for new functionality

4. Update documentation as needed

5. Commit with clear messages:
   ```bash
   git commit -m "feat: add new feature for X"
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Coding Standards

### Python Style Guide

- Follow PEP 8
- Use Black for formatting:
  ```bash
  black src/ tests/
  ```
- Use type hints where possible
- Maximum line length: 88 characters

### Code Quality

- Run linting:
  ```bash
  flake8 src/ tests/
  pylint src/
  mypy src/
  ```

### Docstrings

Use Google-style docstrings:

```python
def process_application(data: dict) -> dict:
    """Process a single application through the AI model.
    
    Args:
        data: Dictionary containing application data with required fields.
        
    Returns:
        Dictionary with prediction results and confidence scores.
        
    Raises:
        ValueError: If required fields are missing.
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_processor.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source code structure
- Use descriptive test names
- Include edge cases
- Mock external dependencies

Example:
```python
def test_process_application_valid_input():
    """Test processing with valid application data."""
    processor = ApplicationProcessor()
    result = processor.process_application(VALID_APPLICATION_DATA)
    
    assert result['success'] is True
    assert 'predicted_quartile' in result
    assert result['confidence'] >= 0 and result['confidence'] <= 100
```

## Documentation

### Code Documentation

- Add docstrings to all public functions/classes
- Include type hints
- Document exceptions
- Provide usage examples

### Project Documentation

- Update relevant `.md` files
- Add new documentation to `docs/`
- Include examples and diagrams
- Keep the README current

## Submitting Changes

### Pull Request Process

1. Update your fork:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. Rebase your feature branch:
   ```bash
   git checkout feature/your-feature-name
   git rebase main
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request on GitHub

### Pull Request Guidelines

- Clear title and description
- Reference related issues
- Include test results
- Update documentation
- Request review from maintainers

### Commit Message Format

Follow the Conventional Commits specification:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Example:
```
feat(processor): add confidence score calibration

Implement isotonic regression for better probability calibration
in the cascade classifier stages.

Closes #123
```

## Questions?

Feel free to:
- Open an issue for questions
- Join our discussions
- Email: admissions-ai@rush.edu

Thank you for contributing to making medical admissions more fair and efficient!