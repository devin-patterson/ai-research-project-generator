# Linting and Security Issue Resolution Guide

This guide provides detailed instructions for identifying and fixing linting and security issues in the AI Research Project Generator.

## Table of Contents

- [Overview](#overview)
- [Tools Used](#tools-used)
- [Running Checks Locally](#running-checks-locally)
- [Common Issues and Fixes](#common-issues-and-fixes)
- [CI/CD Integration](#cicd-integration)
- [Best Practices](#best-practices)

## Overview

The project uses multiple tools to ensure code quality and security:

- **Ruff**: Fast Python linter and formatter
- **Bandit**: Security vulnerability scanner
- **MyPy**: Static type checker
- **pip-audit**: Dependency vulnerability scanner

All checks must pass before code can be merged to the main branch.

## Tools Used

### 1. Ruff (Linting & Formatting)

**Purpose**: Fast Python linter and code formatter that replaces multiple tools (flake8, isort, black)

**Configuration**: `pyproject.toml`
```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
ignore = ["E501"]  # Line too long (handled by formatter)
```

### 2. Bandit (Security Scanner)

**Purpose**: Identifies common security issues in Python code

**Configuration**: `pyproject.toml`
```toml
[tool.bandit]
exclude_dirs = ["tests", ".venv", "venv"]
skips = ["B101"]  # Skip assert warnings in tests
```

### 3. MyPy (Type Checker)

**Purpose**: Static type checking for Python

**Configuration**: `pyproject.toml`
```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

### 4. pip-audit (Dependency Scanner)

**Purpose**: Scans Python dependencies for known security vulnerabilities

**Configuration**: No configuration needed, uses PyPI vulnerability database

## Running Checks Locally

### Prerequisites

```bash
# Ensure you have uv installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### Run All Checks

```bash
# Run all checks at once
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run bandit -r src/ -c pyproject.toml
uv run pip-audit
```

### Run Individual Checks

#### Ruff Linting

```bash
# Check for linting issues
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Check specific files
uv run ruff check src/ai_research_generator/api/
```

#### Ruff Formatting

```bash
# Check formatting (doesn't modify files)
uv run ruff format --check .

# Format files
uv run ruff format .

# Format specific files
uv run ruff format src/ai_research_generator/api/main.py
```

#### Bandit Security Scan

```bash
# Run security scan
uv run bandit -r src/ -c pyproject.toml

# Generate JSON report
uv run bandit -r src/ -c pyproject.toml -f json -o bandit-report.json

# Scan specific directories
uv run bandit -r src/ai_research_generator/api/ -c pyproject.toml
```

#### MyPy Type Checking

```bash
# Run type checking
uv run mypy .

# Check specific module
uv run mypy src/ai_research_generator/api/
```

#### pip-audit Vulnerability Scan

```bash
# Scan dependencies
uv run pip-audit

# Generate JSON report
uv run pip-audit --format json --output audit-report.json
```

## Common Issues and Fixes

### Ruff Linting Issues

#### Issue: Unused imports

```python
# ❌ Bad
import os
import sys
from typing import Dict

def hello():
    print("Hello")
```

**Fix**: Remove unused imports
```python
# ✅ Good
def hello():
    print("Hello")
```

**Auto-fix**: `uv run ruff check --fix .`

#### Issue: Import order

```python
# ❌ Bad
from typing import Dict
import os
from mymodule import something
```

**Fix**: Organize imports (standard library, third-party, local)
```python
# ✅ Good
import os
from typing import Dict

from mymodule import something
```

**Auto-fix**: `uv run ruff check --fix .`

#### Issue: Line too long

```python
# ❌ Bad
def very_long_function_name(parameter1, parameter2, parameter3, parameter4, parameter5, parameter6):
    pass
```

**Fix**: Break into multiple lines
```python
# ✅ Good
def very_long_function_name(
    parameter1,
    parameter2,
    parameter3,
    parameter4,
    parameter5,
    parameter6,
):
    pass
```

**Auto-fix**: `uv run ruff format .`

### Bandit Security Issues

#### Issue: B104 - Hardcoded bind to all interfaces

```python
# ❌ Bad - Security risk: binds to all network interfaces
uvicorn.run(
    "app.main:app",
    host="0.0.0.0",  # Allows external connections
    port=8000,
)
```

**Fix**: Bind to localhost only
```python
# ✅ Good - Only accepts local connections
uvicorn.run(
    "app.main:app",
    host="127.0.0.1",  # Localhost only
    port=8000,
)
```

**When to use 0.0.0.0**: Only in production with proper firewall/proxy configuration

#### Issue: B311 - Random for security purposes

```python
# ❌ Bad - Using random for security
import random
token = random.randint(1000, 9999)
```

**Fix**: Use secrets module for security
```python
# ✅ Good - Cryptographically secure random
import secrets
token = secrets.randbelow(9000) + 1000
```

**Exception**: If using `random` for non-security purposes (e.g., jitter, sampling), add `# nosec: B311`
```python
# ✅ Acceptable for non-security use
import random
jitter = random.uniform(0, 0.25)  # nosec: B311
```

#### Issue: B405/B314 - XML parsing vulnerabilities

```python
# ❌ Bad - Vulnerable to XML attacks
import xml.etree.ElementTree as ET
root = ET.fromstring(untrusted_xml)
```

**Fix Option 1**: Use defusedxml (recommended)
```python
# ✅ Good - Safe XML parsing
import defusedxml.ElementTree as ET
root = ET.fromstring(untrusted_xml)
```

**Fix Option 2**: If source is trusted, add nosec comment
```python
# ✅ Acceptable for trusted sources only
import xml.etree.ElementTree as ET  # nosec: B405
root = ET.fromstring(trusted_xml)  # nosec: B314
```

#### Issue: B608 - SQL injection

```python
# ❌ Bad - SQL injection vulnerability
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
```

**Fix**: Use parameterized queries
```python
# ✅ Good - Safe from SQL injection
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))
```

#### Issue: B201 - Flask debug mode

```python
# ❌ Bad - Debug mode in production
app.run(debug=True)
```

**Fix**: Disable debug in production
```python
# ✅ Good - Debug controlled by environment
import os
app.run(debug=os.getenv("DEBUG", "false").lower() == "true")
```

### MyPy Type Issues

#### Issue: Missing type annotations

```python
# ❌ Bad
def process_data(data):
    return data.upper()
```

**Fix**: Add type annotations
```python
# ✅ Good
def process_data(data: str) -> str:
    return data.upper()
```

#### Issue: Incompatible types

```python
# ❌ Bad
def get_count() -> int:
    return "5"  # Type error: str vs int
```

**Fix**: Return correct type
```python
# ✅ Good
def get_count() -> int:
    return 5
```

#### Issue: Optional type not handled

```python
# ❌ Bad
def get_name(user: Optional[dict]) -> str:
    return user["name"]  # Error: user might be None
```

**Fix**: Handle None case
```python
# ✅ Good
def get_name(user: Optional[dict]) -> str:
    if user is None:
        return "Unknown"
    return user["name"]
```

### pip-audit Vulnerabilities

#### Issue: Vulnerable dependency

```
Found 2 known vulnerabilities in 1 package
Name    Version ID             Fix Versions
------- ------- -------------- ------------
urllib3 1.26.5  PYSEC-2021-108 1.26.6
```

**Fix**: Update the vulnerable package
```bash
# Update specific package
uv pip install --upgrade urllib3

# Or update all packages
uv sync --upgrade
```

#### Issue: Transitive dependency vulnerability

**Fix**: Update the parent package or add explicit version constraint
```toml
# In pyproject.toml
[project]
dependencies = [
    "requests>=2.28.0",  # This will pull in safe urllib3
]
```

## CI/CD Integration

### GitHub Actions Workflow

The project uses GitHub Actions for automated checks. See `.github/workflows/`:

- **ci.yml**: Runs linting, formatting, type checking, and tests
- **security.yml**: Runs Bandit, pip-audit, CodeQL, and secret scanning

### Workflow Triggers

Checks run on:
- Every push to any branch
- Every pull request
- Manual workflow dispatch

### Required Checks

The following checks must pass before merging:

1. ✅ Lint & Format Check (Ruff)
2. ✅ Type Check (MyPy)
3. ✅ Tests (Python 3.10, 3.11, 3.12)
4. ✅ Bandit Security Linter
5. ✅ Vulnerability Scan (pip-audit)
6. ✅ Secret Scanning
7. ✅ CodeQL Analysis
8. ✅ Dependency Review

### Viewing CI Results

```bash
# View PR checks
gh pr checks <PR_NUMBER>

# View specific workflow run
gh run view <RUN_ID> --log

# View latest run for current branch
gh run list --branch=$(git branch --show-current) --limit=1
```

## Best Practices

### 1. Run Checks Before Committing

```bash
# Create a pre-commit script
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
EOF

chmod +x .git/hooks/pre-commit
```

### 2. Use Editor Integration

**VS Code** (`.vscode/settings.json`):
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "ruff",
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true
}
```

**PyCharm**:
- Settings → Tools → External Tools → Add Ruff
- Settings → Editor → Inspections → Enable MyPy

### 3. Suppress Warnings Carefully

Only suppress warnings when you understand the issue and have a valid reason:

```python
# ✅ Good - Clear reason for suppression
import random
delay = random.uniform(0, 1)  # nosec: B311 - Non-security jitter

# ❌ Bad - No explanation
some_risky_code()  # nosec
```

### 4. Keep Dependencies Updated

```bash
# Check for outdated packages
uv pip list --outdated

# Update all packages
uv sync --upgrade

# Run security scan after updates
uv run pip-audit
```

### 5. Review Security Reports

```bash
# Generate comprehensive security report
uv run bandit -r src/ -c pyproject.toml -f json -o bandit-report.json
uv run pip-audit --format json --output audit-report.json

# Review reports
cat bandit-report.json | jq '.results[] | {file: .filename, issue: .issue_text}'
cat audit-report.json | jq '.vulnerabilities[] | {package: .name, vulnerability: .id}'
```

### 6. Document Security Decisions

When adding `# nosec` comments, document why:

```python
# Parse XML from trusted arXiv API
# nosec: B405, B314 - XML source is trusted academic database
import xml.etree.ElementTree as ET
root = ET.fromstring(response.text)
```

## Troubleshooting

### Issue: Ruff not found

```bash
# Reinstall development dependencies
uv sync
```

### Issue: Conflicting formatters

```bash
# Remove other formatters
uv pip uninstall black autopep8 yapf

# Use only Ruff
uv run ruff format .
```

### Issue: MyPy import errors

```bash
# Install type stubs
uv pip install types-requests types-PyYAML

# Or ignore missing imports
# In pyproject.toml:
[tool.mypy]
ignore_missing_imports = true
```

### Issue: Bandit false positives

```python
# Suppress specific issue with explanation
risky_function()  # nosec: B123 - Explanation here
```

### Issue: CI passes locally but fails in CI

```bash
# Ensure same Python version
python --version  # Should match CI (3.10+)

# Clear cache and reinstall
rm -rf .venv
uv sync

# Run exact CI commands
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run bandit -r src/ -c pyproject.toml
```

## Additional Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [pip-audit Documentation](https://pypi.org/project/pip-audit/)
- [OWASP Python Security](https://owasp.org/www-project-python-security/)

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/devin-patterson/ai-research-project-generator/issues)
2. Review CI logs: `gh run view <RUN_ID> --log`
3. Ask in the project's discussion forum
4. Consult the official documentation for each tool
