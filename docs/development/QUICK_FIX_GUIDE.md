# Quick Fix Guide for Common Issues

This guide provides rapid solutions for the most common development and CI/CD issues in the AI Research Project Generator.

## 🚀 Quick Navigation

- [Import Errors](#import-errors) - 2 min fix
- [Test Failures](#test-failures) - 3 min fix  
- [Linting Issues](#linting-issues) - 1 min fix
- [Security Issues](#security-issues) - 5 min fix
- [Build Issues](#build-issues) - 2 min fix
- [CI Failures](#ci-failures) - 5 min fix

---

## 🔧 Import Errors

### Issue: `ModuleNotFoundError: No module named 'app'`

**Quick Fix (30 seconds):**
```bash
# Find and replace all 'from app.' imports
find . -name "*.py" -exec sed -i '' 's/from app\./from src.ai_research_generator./g' {} \;
```

**Manual Fix (2 minutes):**
```python
# ❌ Replace these
from app.core.config import Settings
from app.schemas.research import ResearchRequest
from app.workflows.agents import TopicAnalysis

# ✅ With these
from src.ai_research_generator.core.config import Settings
from src.ai_research_generator.models.schemas.research import ResearchRequest
from src.ai_research_generator.workflows.agents import TopicAnalysis
```

### Issue: Relative Import Errors

**Quick Fix (1 minute):**
```python
# ❌ Wrong relative imports
from app.core.config import Settings
from app.schemas.research import ResearchRequest

# ✅ Correct relative imports  
from ..core.config import Settings
from ..models.schemas.research import ResearchRequest
```

### Issue: Legacy Module Imports

**Quick Fix (1 minute):**
```python
# ❌ Old imports
from llm_provider import LLMProvider
from academic_search import UnifiedAcademicSearch

# ✅ New relative imports
from .llm_provider import LLMProvider
from .academic_search import UnifiedAcademicSearch
```

---

## 🧪 Test Failures

### Issue: Test Collection Errors

**Quick Fix (2 minutes):**
```bash
# Update test imports
sed -i '' 's/from app\./from src.ai_research_generator./g' tests/*.py
sed -i '' 's/from app\.schemas\./from src.ai_research_generator.models.schemas./g' tests/*.py
```

### Issue: Mock Path Errors in Tests

**Quick Fix (1 minute):**
```python
# ❌ Wrong patch paths
with patch("app.workflows.agents.create_agent"):
with patch("app.schemas.research.ResearchRequest"):

# ✅ Correct patch paths
with patch("src.ai_research_generator.workflows.agents.create_agent"):
with patch("src.ai_research_generator.models.schemas.research.ResearchRequest"):
```

### Issue: Async Test Errors

**Quick Fix (30 seconds):**
```python
# ❌ Missing async marker
def test_async_function():
    await some_async_function()

# ✅ Add async marker
@pytest.mark.asyncio
async def test_async_function():
    await some_async_function()
```

**Run Tests:**
```bash
uv run pytest tests/test_failing.py -v
```

---

## ✨ Linting Issues

### Issue: Ruff Formatting Errors

**Quick Fix (30 seconds):**
```bash
# Auto-format all files
uv run ruff format .

# Check specific files
uv run ruff format src/ai_research_generator/api/main.py
```

### Issue: Ruff Linting Errors

**Quick Fix (30 seconds):**
```bash
# Auto-fix linting issues
uv run ruff check --fix .

# Check specific file
uv run ruff check src/ai_research_generator/api/main.py
```

### Issue: Common Linting Errors

**Quick Fixes (1 minute):**

```python
# ❌ Unused variable
result = some_function()  # F841

# ✅ Fix
result = some_function()
_ = result  # Or use underscore

# ❌ Import order
import os
from typing import Dict
from mymodule import func

# ✅ Fix
import os
from typing import Dict

from mymodule import func

# ❌ Line too long
very_long_variable_name = "this is a very long string that exceeds the line length limit"

# ✅ Fix
very_long_variable_name = (
    "this is a very long string that exceeds the line length limit"
)
```

---

## 🔒 Security Issues

### Issue: Bandit B104 - Hardcoded Bind

**Quick Fix (1 minute):**
```python
# ❌ Before
uvicorn.run("app.main:app", host="0.0.0.0", port=8000)

# ✅ After
uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
```

### Issue: Bandit B311 - Random Usage

**Quick Fix (30 seconds):**
```python
# ❌ Before
jitter = random.uniform(0, 0.25)

# ✅ After
jitter = random.uniform(0, 0.25)  # nosec: B311 - Non-security use
```

### Issue: Bandit B405/B314 - XML Parsing

**Quick Fix (1 minute):**
```python
# ❌ Before
import xml.etree.ElementTree as ET
root = ET.fromstring(xml_data)

# ✅ After
import xml.etree.ElementTree as ET  # nosec: B405
root = ET.fromstring(xml_data)  # nosec: B314 - Trusted source
```

**Run Security Check:**
```bash
uv run bandit -r src/ -c pyproject.toml
```

---

## 🏗️ Build Issues

### Issue: Hatchling Build Error

**Quick Fix (30 seconds):**
Add to `pyproject.toml`:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/ai_research_generator"]
```

### Issue: Dependency Conflicts

**Quick Fix (1 minute):**
```bash
# Clean reinstall
rm -rf .venv
uv sync

# Check for conflicts
uv pip check

# Update specific package
uv pip install --upgrade package-name
```

### Issue: Missing Dependencies

**Quick Fix (30 seconds):**
```bash
# Add missing dependency
uv add package-name

# Add dev dependency
uv add --dev package-name
```

---

## 🔄 CI Failures

### Issue: CI Tests Fail Locally

**Quick Fix (2 minutes):**
```bash
# Sync with CI environment
git checkout <COMMIT_SHA>
uv sync

# Run exact CI commands
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest tests/
uv run bandit -r src/ -c pyproject.toml
```

### Issue: CI Cache Problems

**Quick Fix (1 minute):**
```bash
# Clear GitHub Actions cache
gh api repos/:owner/:repo/actions/caches --jq '.actions_caches[].id' | xargs -I {} gh api --method DELETE repos/:owner/:repo/actions/caches/{}

# Or wait for cache to expire (24 hours)
```

### Issue: CI Timeout

**Quick Fix (1 minute):**
Add to `.github/workflows/ci.yml`:
```yaml
jobs:
  test:
    timeout-minutes: 30  # Increase timeout
```

### Issue: CI Permission Errors

**Quick Fix (30 seconds):**
Add to workflow file:
```yaml
permissions:
  contents: read
  checks: write
  pull-requests: write
```

---

## 🚨 Emergency Fixes

### All Tests Failing

**Emergency Fix (2 minutes):**
```bash
# Reset to last working commit
git log --oneline -5
git reset --hard <WORKING_COMMIT>

# Force push (use with caution)
git push --force-with-lease origin branch-name
```

### Import System Completely Broken

**Emergency Fix (3 minutes):**
```bash
# Check package structure
find src/ -name "__init__.py" | head -10

# Fix main imports
echo 'from .api.main import app' > src/ai_research_generator/__init__.py

# Test import
uv run python -c "from src.ai_research_generator.api.main import app; print('✅ Import works')"
```

### Security Scanner Fails

**Emergency Fix (2 minutes):**
```bash
# Run security scan locally
uv run bandit -r src/ -c pyproject.toml

# Quick fix common issues
sed -i '' 's/host="0.0.0.0"/host="127.0.0.1"/g' src/**/*.py
sed -i '' 's/random\./random./g' src/**/*.py | head -5  # Check if needs nosec

# Add nosec comments where appropriate
```

---

## 📋 One-Command Fixes

### Fix All Import Issues
```bash
find . -name "*.py" -not -path "./.venv/*" -exec sed -i '' 's/from app\./from src.ai_research_generator./g' {} \;
find . -name "*.py" -not -path "./.venv/*" -exec sed -i '' 's/from app\.schemas\./from src.ai_research_generator.models.schemas./g' {} \;
```

### Fix All Formatting Issues
```bash
uv run ruff format .
uv run ruff check --fix .
```

### Fix All Test Imports
```bash
sed -i '' 's/from app\./from src.ai_research_generator./g' tests/*.py
sed -i '' 's/from app\.schemas\./from src.ai_research_generator.models.schemas./g' tests/*.py
```

### Fix All Security Issues (Common)
```bash
sed -i '' 's/host="0.0.0.0"/host="127.0.0.1"/g' src/**/*.py
```

---

## 🔍 Verification Commands

### After Import Fixes
```bash
uv run python -c "from src.ai_research_generator.api.main import app; print('✅ API import works')"
uv run python -c "from src.ai_research_generator.models.schemas.research import ResearchRequest; print('✅ Models import works')"
```

### After Test Fixes
```bash
uv run pytest tests/ --collect-only | head -10
uv run pytest tests/test_api.py::TestHealthEndpoint::test_health_check -v
```

### After Linting Fixes
```bash
uv run ruff check .
uv run ruff format --check .
```

### After Security Fixes
```bash
uv run bandit -r src/ -c pyproject.toml
```

---

## 📞 Get Help

### Still Stuck?

1. **Check detailed guides**:
   - [Linting and Security Guide](LINTING_AND_SECURITY.md)
   - [CI/CD Troubleshooting Guide](CI_TROUBLESHOOTING.md)

2. **Run diagnostics**:
```bash
# Check environment
uv run python -c "import sys; print('Python:', sys.version)"
uv run python -c "import sys; print('Path:', sys.path)"

# Check imports
uv run python -c "import src.ai_research_generator; print('✅ Package imports work')"
```

3. **Get specific error help**:
```bash
# Search for similar issues
gh issue list --search "import error" --limit=5

# Ask in discussions
gh discussion create --title "Help with import error" --body "Describe your issue here"
```

### Quick Reference Commands

```bash
# Environment check
uv --version && python --version

# Dependencies check
uv pip list | head -10

# Test single file
uv run pytest tests/test_api.py -v

# Security check
uv run bandit -r src/ -c pyproject.toml | head -20

# CI status
gh pr checks <PR_NUMBER>
```

---

## 💡 Prevention Tips

### Before Committing
```bash
# Run full check suite
uv run ruff check . && uv run ruff format --check . && uv run mypy . && uv run pytest tests/ && uv run bandit -r src/ -c pyproject.toml
```

### IDE Setup
- Install Ruff extension
- Enable format-on-save
- Set up import sorting
- Configure type checking

### Pre-commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/sh
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest tests/ --tb=short
```

---

*Last updated: 2026-01-31*  
*For detailed troubleshooting, see the full guides in this directory.*
