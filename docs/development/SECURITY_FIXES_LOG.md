# Security Fixes Log

This document tracks all security-related fixes applied to the AI Research Project Generator, including the reasoning behind each fix and how to prevent similar issues in the future.

## 📋 Table of Contents

- [Recent Fixes](#recent-fixes)
- [Fix Categories](#fix-categories)
- [Prevention Strategies](#prevention-strategies)
- [Monitoring](#monitoring)
- [References](#references)

## 🛠️ Recent Fixes

### Fix #1: Bandit Security Linter Issues
**Date**: 2026-01-31  
**Commit**: `b4d93b8`  
**Pull Request**: #5

#### Issues Fixed

1. **B104 - Hardcoded Bind to All Interfaces**
   - **File**: `src/ai_research_generator/api/main.py:152`
   - **Issue**: `host="0.0.0.0"` allows external connections
   - **Risk**: Medium - Exposes service to network attacks
   - **Fix**: Changed to `host="127.0.0.1"` for localhost only

```python
# ❌ Before
uvicorn.run(
    "app.main:app",
    host="0.0.0.0",  # Security risk
    port=8000,
)

# ✅ After
uvicorn.run(
    "app.main:app",
    host="127.0.0.1",  # Localhost only
    port=8000,
)
```

**Reasoning**: 
- Development servers should only accept local connections
- Production deployments should use reverse proxies (nginx, traefik) for external access
- Prevents direct exposure to network attacks

**Prevention**:
- Use environment variables for host configuration
- Document when `0.0.0.0` is acceptable (production with proxy)
- Add security review checklist for network configurations

2. **B311 - Standard Pseudo-Random Generators**
   - **File**: `src/ai_research_generator/core/retry.py:63`
   - **Issue**: `random.uniform()` used for jitter
   - **Risk**: Low - Not security-critical, but flagged by Bandit
   - **Fix**: Added `# nosec: B311` comment with explanation

```python
# ❌ Before
jitter_amount = delay * random.uniform(0, 0.25)

# ✅ After
jitter_amount = delay * random.uniform(0, 0.25)  # nosec: B311 - Non-security use for jitter
```

**Reasoning**:
- Random usage for retry jitter is not security-critical
- `secrets` module would be overkill and slower
- Explicit nosec comment documents the decision

**Prevention**:
- Document acceptable use cases for `random` vs `secrets`
- Add code comments explaining security decisions
- Include in security review guidelines

3. **B405/B314 - XML Parsing Vulnerabilities**
   - **File**: `src/ai_research_generator/legacy/academic_search.py:710-712`
   - **Issue**: `xml.etree.ElementTree` used for parsing
   - **Risk**: Medium - XML attacks possible
   - **Fix**: Added `# nosec` comments for trusted source

```python
# ❌ Before
import xml.etree.ElementTree as ET
root = ET.fromstring(response.text)

# ✅ After
import xml.etree.ElementTree as ET  # nosec: B405
root = ET.fromstring(response.text)  # nosec: B314 - Trusted arXiv API source
```

**Reasoning**:
- XML source is trusted arXiv academic database
- `defusedxml` would add dependency for minimal benefit
- Explicit nosec comments document the trust decision

**Prevention**:
- Evaluate XML source trustworthiness before parsing
- Consider `defusedxml` for untrusted XML sources
- Document trusted data sources in code comments

### Fix #2: Hatchling Build Configuration
**Date**: 2026-01-31  
**Commit**: `b4d93b8`  
**Pull Request**: #5

#### Issue Fixed

**Build System Configuration Missing**
- **File**: `pyproject.toml`
- **Issue**: Hatchling couldn't find packages in src/ layout
- **Risk**: Low - Build failure, not security
- **Fix**: Added hatchling build configuration

```toml
# ✅ Added
[tool.hatch.build.targets.wheel]
packages = ["src/ai_research_generator"]
```

**Reasoning**:
- Modern Python projects use src/ layout for better packaging
- Hatchling needs explicit package specification for src/ structure
- Prevents build failures in CI/CD pipelines

**Prevention**:
- Use standard project templates for new projects
- Include build configuration in project setup checklist
- Test package builds in CI/CD

## 📊 Fix Categories

### Network Security (B104)
- **Frequency**: Low
- **Impact**: Medium to High
- **Common Locations**: API servers, web applications
- **Prevention**: Environment-based configuration, security review

### Cryptographic Security (B311)
- **Frequency**: Medium
- **Impact**: Low to Medium
- **Common Locations**: Token generation, random functions
- **Prevention**: Use `secrets` for security, document exceptions

### Input Validation (B405/B314)
- **Frequency**: Medium
- **Impact**: Medium to High
- **Common Locations**: XML/JSON parsing, file processing
- **Prevention**: Use safe parsers, validate sources, nosec comments

### Build Security
- **Frequency**: Low
- **Impact**: Low
- **Common Locations**: pyproject.toml, setup.py
- **Prevention**: Standard project templates, build testing

## 🛡️ Prevention Strategies

### 1. Code Review Checklist

#### Network Security
- [ ] Check for hardcoded `0.0.0.0` bindings
- [ ] Verify environment variable usage for network config
- [ ] Review firewall and proxy configurations

#### Cryptographic Security
- [ ] Identify all `random` usage
- [ ] Verify `secrets` module for security-critical operations
- [ ] Check for hardcoded secrets or keys

#### Input Validation
- [ ] Review XML/JSON parsing locations
- [ ] Verify input sanitization
- [ ] Check for SQL injection vulnerabilities

#### Build Security
- [ ] Verify package configuration
- [ ] Check dependency security
- [ ] Review build scripts

### 2. Automated Prevention

#### Pre-commit Hooks
```bash
#!/bin/sh
# .git/hooks/pre-commit
uv run bandit -r src/ -c pyproject.toml
uv run ruff check .
uv run ruff format --check .
```

#### CI/CD Integration
```yaml
# .github/workflows/security.yml
- name: Run Bandit Security Scan
  run: |
    uv run bandit -r src/ -c pyproject.toml -f json -o bandit-report.json
    # Fail on high/medium severity issues
```

#### IDE Integration
```json
// .vscode/settings.json
{
  "python.linting.banditEnabled": true,
  "python.linting.banditArgs": ["-r", "src/", "-c", "pyproject.toml"]
}
```

### 3. Development Guidelines

#### Network Configuration
```python
# ✅ Good practice
import os
from typing import Optional

def get_host() -> str:
    """Get appropriate host based on environment."""
    if os.getenv("ENVIRONMENT") == "production":
        return "0.0.0.0"  # Behind proxy/firewall
    return "127.0.0.1"  # Development

uvicorn.run(
    "app.main:app",
    host=get_host(),
    port=int(os.getenv("PORT", "8000")),
)
```

#### Random Usage Guidelines
```python
# ✅ Security-critical - Use secrets
import secrets
token = secrets.token_urlsafe(32)

# ✅ Non-security - Use random with nosec
import random
jitter = random.uniform(0, 0.25)  # nosec: B311 - Retry jitter

# ✅ Document decision
def generate_test_data():
    """Generate test data using random - acceptable for testing."""
    return random.randint(1, 100)  # nosec: B311 - Test data only
```

#### XML Parsing Guidelines
```python
# ✅ Trusted source with documentation
def parse_arxiv_response(xml_data: str):
    """Parse XML from trusted arXiv API.
    
    Note: arXiv is a trusted academic database, so standard
    ElementTree is acceptable. nosec: B405, B314
    """
    import xml.etree.ElementTree as ET  # nosec: B405
    return ET.fromstring(xml_data)  # nosec: B314

# ✅ Untrusted source - Use defusedxml
def parse_user_upload(xml_data: str):
    """Parse XML from untrusted user upload."""
    import defusedxml.ElementTree as ET
    return ET.fromstring(xml_data)
```

## 📈 Monitoring

### Security Metrics

#### Bandit Metrics
```bash
# Generate security report
uv run bandit -r src/ -c pyproject.toml -f json -o bandit-report.json

# Track metrics over time
cat bandit-report.json | jq '.metrics._totals'
```

#### Trend Analysis
- **High/Medium Issues**: Should be 0
- **Low Issues**: Document and justify
- **New Issues**: Investigate immediately

### Alert Thresholds
```yaml
# .github/workflows/security-alerts.yml
- name: Check Security Issues
  run: |
    HIGH=$(cat bandit-report.json | jq '.metrics._totals.SEVERITY.HIGH')
    MEDIUM=$(cat bandit-report.json | jq '.metrics._totals.SEVERITY.MEDIUM')
    
    if [ "$HIGH" -gt 0 ] || [ "$MEDIUM" -gt 0 ]; then
      echo "❌ Security issues found: High=$HIGH, Medium=$MEDIUM"
      exit 1
    fi
```

### Regular Reviews
- **Weekly**: Security scan results review
- **Monthly**: Dependency vulnerability check
- **Quarterly**: Security architecture review
- **Annually**: Third-party security audit

## 🔧 Tools and Configuration

### Bandit Configuration
```toml
# pyproject.toml
[tool.bandit]
exclude_dirs = ["tests", ".venv", "venv"]
skips = ["B101"]  # Skip assert warnings in tests
tests = ["B201", "B301"]  # Additional tests for security

[tool.bandit.assert_used]
skips = ["B101"]  # Skip assert in tests
```

### Security Tools Integration
```bash
# Install security tools
uv add --dev bandit[toml] safety
uv add --dev semgrep  # Optional: Advanced security scanning

# Run comprehensive security check
uv run bandit -r src/ -c pyproject.toml
uv run safety check
# uv run semgrep --config=security  # Optional
```

### VS Code Security Extensions
- **Bandit**: Python security linter
- **Safety**: Dependency vulnerability checker
- **GitLens**: Security-focused code review

## 📚 References

### Security Resources
- [OWASP Python Security](https://owasp.org/www-project-python-security/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Python Security Best Practices](https://docs.python.org/3/library/security.html)

### Project Resources
- [Project Security Policy](../../SECURITY.md) (TODO)
- [Vulnerability Disclosure](../../SECURITY.md) (TODO)
- [Security Contact Information](../../SECURITY.md) (TODO)

## 🆘 Reporting Security Issues

### Responsible Disclosure
If you discover a security vulnerability:

1. **Do not** create a public issue
2. **Email**: security@project-domain.com
3. **Include**: Detailed description, reproduction steps, impact assessment
4. **Response time**: We aim to respond within 48 hours

### Security Contact
- **Email**: security@project-domain.com
- **PGP Key**: Available on request
- **Bug Bounty**: Not currently available

## 📝 Maintenance

### Regular Tasks
- [ ] Weekly security scan review
- [ ] Monthly dependency updates
- [ ] Quarterly security documentation updates
- [ ] Annual security architecture review

### Documentation Updates
- Update this log for all security fixes
- Review and update prevention strategies
- Maintain current security best practices
- Update tool configurations as needed

---

*Last updated: 2026-01-31*  
*Next review: 2026-02-28*
