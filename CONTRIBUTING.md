# Contributing to AI Research Project Generator

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-research-project-generator.git
   cd ai-research-project-generator
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/depatter/ai-research-project-generator.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai/) (optional, for LLM features)

### Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --all-extras --dev

# Set up pre-commit hooks (optional but recommended)
uv run pre-commit install
```

### Environment Configuration

Copy the example environment file and configure:

```bash
cp .env.example .env
# Edit .env with your settings
```

### Running the Application

```bash
# Start the API server
uv run uvicorn app.main:app --reload --port 8000

# Or use the CLI
uv run python main.py --topic "Your research topic"
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-search-provider`
- `fix/llm-connection-timeout`
- `docs/update-api-reference`
- `refactor/simplify-validation`

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

Examples:
```
feat(api): add endpoint for batch research generation
fix(llm): handle timeout errors gracefully
docs(readme): update installation instructions
```

## Code Style

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use [PEP 257](https://peps.python.org/pep-0257/) for docstrings
- Maximum line length: 100 characters
- Use type hints for all function signatures

### Formatting and Linting

We use `ruff` for linting and formatting:

```bash
# Check for linting issues
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Checking

```bash
uv run mypy app/ --ignore-missing-imports
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_api.py

# Run tests matching a pattern
uv run pytest -k "test_research"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names
- Include both positive and negative test cases

Example:
```python
import pytest
from app.schemas.research import ResearchRequest

def test_research_request_valid():
    """Test that valid research request is accepted."""
    request = ResearchRequest(
        topic="Impact of AI on education",
        research_question="How does AI affect learning outcomes?",
        research_type="systematic_review",
    )
    assert request.topic == "Impact of AI on education"

def test_research_request_topic_too_short():
    """Test that short topics are rejected."""
    with pytest.raises(ValueError):
        ResearchRequest(
            topic="AI",  # Too short
            research_question="How does AI affect learning?",
        )
```

## Submitting Changes

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** and commit them

4. **Run tests and linting**:
   ```bash
   uv run ruff check .
   uv run pytest
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

### Pull Request Guidelines

- Fill out the PR template completely
- Link related issues
- Ensure CI passes
- Request review from maintainers
- Be responsive to feedback

## Reporting Issues

### Bug Reports

When reporting bugs, include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

### Feature Requests

When requesting features, include:
- Problem statement
- Proposed solution
- Use cases
- Alternatives considered

## Questions?

If you have questions, feel free to:
- Open a GitHub Discussion
- Create an issue with the `question` label
- Reach out to the maintainers

Thank you for contributing! 🎉
