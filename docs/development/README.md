# Development Documentation

This directory contains comprehensive documentation for developers working on the AI Research Project Generator.

## 📚 Documentation Structure

```
docs/development/
├── README.md                    # This file - Development overview
├── LINTING_AND_SECURITY.md      # Linting and security issue resolution
├── CI_TROUBLESHOOTING.md        # CI/CD troubleshooting guide
└── CONTRIBUTING.md              # Contributing guidelines (TODO)
```

## 🚀 Quick Start for Developers

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager
- Git
- GitHub account (for contributions)

### Setup

```bash
# Clone the repository
git clone https://github.com/devin-patterson/ai-research-project-generator.git
cd ai-research-project-generator

# Install dependencies
uv sync

# Run tests to verify setup
uv run pytest tests/
```

### Development Workflow

1. **Create feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes**
```bash
# Edit code
# Run tests locally
uv run pytest tests/
# Check linting and formatting
uv run ruff check .
uv run ruff format --check .
# Run security checks
uv run bandit -r src/ -c pyproject.toml
```

3. **Commit and push**
```bash
git add .
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
```

4. **Create Pull Request**
- Go to GitHub and create PR
- Ensure all CI checks pass
- Request code review

## 📋 Development Guidelines

### Code Quality Standards

- **Linting**: All code must pass Ruff checks
- **Formatting**: Use Ruff formatter (auto-formatted on save)
- **Type Checking**: All code must pass MyPy checks
- **Testing**: Maintain >80% test coverage
- **Security**: Pass Bandit security scans

### Architecture Principles

- **Modular Design**: Separate concerns into distinct modules
- **Clean Architecture**: Use dependency injection and interfaces
- **Type Safety**: Use Pydantic models and type hints
- **Async/Await**: Use async patterns for I/O operations
- **Error Handling**: Use structured exception handling

### AI Integration Standards

- **PydanticAI**: Use for type-safe LLM interactions
- **LangGraph**: Use for complex workflow orchestration
- **DSPy**: Use for prompt optimization
- **DeepEval**: Use for LLM evaluation and testing

## 🔧 Development Tools

### Essential Commands

```bash
# Development server
uv run uvicorn src.ai_research_generator.api.main:app --reload

# Run tests
uv run pytest tests/ -v

# Code quality checks
uv run ruff check .
uv run ruff format .
uv run mypy .

# Security checks
uv run bandit -r src/ -c pyproject.toml
uv run pip-audit

# Dependency management
uv sync
uv pip install <package>
uv pip list
```

### IDE Configuration

#### VS Code

Install these extensions:
- Python (Microsoft)
- Ruff (Astral)
- MyPy (ms-python.mypy-type-checker)

VS Code settings (`.vscode/settings.json`):
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "ruff",
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

#### PyCharm

- Enable "Type checker" plugin
- Configure external tools for Ruff
- Set up code style to match project standards

## 🧪 Testing Strategy

### Test Types

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **API Tests**: Test REST API endpoints
4. **LLM Tests**: Test AI model interactions
5. **Golden Set Tests**: Regression testing with known outputs

### Test Organization

```
tests/
├── unit/                    # Unit tests
├── integration/             # Integration tests
├── api/                     # API tests
├── llm/                     # LLM interaction tests
├── golden_set/              # Golden set tests
└── fixtures/                # Test data and fixtures
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_api.py

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run specific test category
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
```

### Writing Tests

```python
import pytest
from src.ai_research_generator.models.schemas.research import ResearchRequest

class TestResearchRequest:
    def test_valid_request(self):
        """Test valid research request creation."""
        request = ResearchRequest(
            topic="AI in Education",
            research_question="How does AI affect learning?",
            research_type="systematic_review",
            academic_level="graduate"
        )
        assert request.topic == "AI in Education"
        assert request.research_type == ResearchType.SYSTEMATIC_REVIEW
```

## 🔒 Security Guidelines

### Security Best Practices

1. **Input Validation**: Always validate user input
2. **Secrets Management**: Never commit secrets to repository
3. **Dependency Security**: Regularly update dependencies
4. **Code Review**: All code must be reviewed
5. **Security Testing**: Run security scans regularly

### Security Tools

- **Bandit**: Static security analysis
- **pip-audit**: Dependency vulnerability scanning
- **CodeQL**: Advanced security analysis
- **Secret Scanning**: GitHub's built-in secret detection

### Common Security Issues

```python
# ❌ Bad - Hardcoded secrets
API_KEY = "sk-1234567890abcdef"

# ✅ Good - Environment variables
import os
API_KEY = os.getenv("API_KEY")

# ❌ Bad - SQL injection vulnerability
query = f"SELECT * FROM users WHERE id = {user_id}"

# ✅ Good - Parameterized queries
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))
```

## 📦 Package Management

### Dependency Management

```bash
# Add new dependency
uv add <package>

# Add development dependency
uv add --dev <package>

# Update dependencies
uv sync --upgrade

# Remove dependency
uv remove <package>

# Check for vulnerabilities
uv run pip-audit
```

### Package Structure

```
src/ai_research_generator/
├── __init__.py
├── api/                     # FastAPI application
├── core/                    # Core business logic
├── models/                  # Pydantic models
├── services/                # Service layer
├── workflows/               # AI workflows
├── agents/                  # AI agents
├── optimization/            # Prompt optimization
└── legacy/                  # Legacy code
```

### Import Guidelines

```python
# ✅ Good - Relative imports within package
from .core.config import Settings
from ..models.schemas.research import ResearchRequest

# ✅ Good - Absolute imports for external packages
from fastapi import FastAPI
from pydantic import BaseModel

# ❌ Bad - Mixed import styles
from app.core.config import Settings  # Old style
```

## 🚀 Deployment

### Environment Setup

```bash
# Production dependencies
uv sync --group production

# Environment variables
export DATABASE_URL="postgresql://..."
export API_KEY="sk-..."
export DEBUG="false"
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install dependencies
COPY pyproject.toml ./
RUN uv sync --frozen

# Copy application
COPY src/ ./src/

# Run application
CMD ["uv", "run", "uvicorn", "src.ai_research_generator.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Monitoring

- **Health Checks**: `/health` endpoint
- **Metrics**: Prometheus metrics (TODO)
- **Logging**: Structured logging with loguru
- **Error Tracking**: Sentry integration (TODO)

## 🤝 Contributing

### Contributing Process

1. **Fork** the repository
2. **Create** feature branch
3. **Make** changes with tests
4. **Ensure** all checks pass
5. **Submit** pull request
6. **Address** review feedback
7. **Merge** when approved

### Code Review Guidelines

- **Functionality**: Does the code work as intended?
- **Testing**: Are tests comprehensive?
- **Style**: Does code follow project standards?
- **Security**: Are there security concerns?
- **Documentation**: Is code well-documented?

### Commit Messages

```
feat: add new feature
fix: resolve bug in API
docs: update documentation
style: fix code formatting
refactor: improve code structure
test: add tests for new feature
security: fix security vulnerability
```

## 📚 Additional Resources

### Documentation

- [Project README](../../README.md)
- [Architecture Guide](../ARCHITECTURE.md)
- [AI Enablement Guide](../ai/AI_ENABLEMENT.md)

### External Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [PydanticAI Documentation](https://pydantic-ai.readthedocs.io/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [DSPy Documentation](https://dspy-docs.vercel.app/)

### Community

- [GitHub Issues](https://github.com/devin-patterson/ai-research-project-generator/issues)
- [GitHub Discussions](https://github.com/devin-patterson/ai-research-project-generator/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/ai-research-project-generator)

## 🆘 Getting Help

If you need help:

1. **Check documentation**: Review existing docs first
2. **Search issues**: Look for similar problems
3. **Ask questions**: Use GitHub discussions
4. **Report bugs**: Create detailed issue with reproduction steps
5. **Contact maintainers**: Use GitHub @ mentions

### Troubleshooting

For common issues, see:
- [Linting and Security Guide](LINTING_AND_SECURITY.md)
- [CI/CD Troubleshooting Guide](CI_TROUBLESHOOTING.md)

---

*Last updated: 2026-01-31*
