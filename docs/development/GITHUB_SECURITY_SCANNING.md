# GitHub Security Scanning: Native vs GitHub Actions

This document explains which security scans are performed natively by GitHub versus those configured via GitHub Actions workflows.

## 📊 Security Scanning Overview

### Native GitHub Security Features

These features are built into GitHub and run automatically without requiring workflow configuration:

#### 1. **Secret Scanning** 🔐
- **Type**: Native GitHub Feature
- **Location**: Repository Settings → Security → Code security and analysis
- **How it works**: Automatically scans commits for known secret patterns
- **Coverage**: 
  - API keys, tokens, credentials
  - Partner patterns (AWS, Azure, Google, etc.)
  - Custom patterns (GitHub Advanced Security)
- **Alerts**: Appear in Security tab → Secret scanning alerts
- **Cost**: Free for public repos, requires GitHub Advanced Security for private repos

#### 2. **Dependabot Alerts** 🔔
- **Type**: Native GitHub Feature
- **Location**: Repository Settings → Security → Code security and analysis
- **How it works**: Monitors dependencies for known vulnerabilities
- **Coverage**:
  - Python (pip, poetry, pipenv)
  - JavaScript/Node.js
  - Ruby, Java, .NET, Go, etc.
- **Alerts**: Appear in Security tab → Dependabot alerts
- **Auto-fix**: Dependabot can create PRs to update vulnerable dependencies
- **Cost**: Free for all repositories

#### 3. **Dependabot Security Updates** 🔄
- **Type**: Native GitHub Feature
- **Location**: Repository Settings → Security → Code security and analysis
- **How it works**: Automatically creates PRs to update vulnerable dependencies
- **Requires**: Dependabot alerts to be enabled
- **Cost**: Free for all repositories

#### 4. **Code Scanning (CodeQL)** 🔍
- **Type**: Hybrid (Native + Actions)
- **Setup Options**:
  - **Default Setup** (Native): GitHub automatically configures CodeQL
  - **Advanced Setup** (Actions): Custom `.github/workflows/codeql.yml`
- **How it works**: Static analysis to find security vulnerabilities
- **Coverage**: 
  - Python, JavaScript, TypeScript, Java, C/C++, C#, Go, Ruby
  - Security vulnerabilities, code quality issues
- **Alerts**: Appear in Security tab → Code scanning alerts
- **Cost**: Free for public repos, requires GitHub Advanced Security for private repos

### GitHub Actions-Based Security Scans

These require explicit workflow configuration in `.github/workflows/`:

#### 1. **Bandit Security Linter** 🛡️
- **Type**: GitHub Actions Workflow
- **File**: `.github/workflows/security.yml`
- **Tool**: Bandit (Python security linter)
- **How it works**: Runs on every push/PR via GitHub Actions
- **Coverage**:
  - Python-specific security issues
  - Hardcoded passwords, SQL injection, etc.
  - Custom security patterns
- **Configuration**: `pyproject.toml` → `[tool.bandit]`
- **Cost**: Free (uses GitHub Actions minutes)

```yaml
# .github/workflows/security.yml
- name: Run Bandit Security Scan
  run: |
    uv run bandit -r src/ -c pyproject.toml
```

#### 2. **Dependency Review** 📦
- **Type**: GitHub Actions Workflow
- **File**: `.github/workflows/security.yml`
- **Action**: `actions/dependency-review-action@v4`
- **How it works**: Reviews dependency changes in PRs
- **Coverage**:
  - New vulnerabilities introduced
  - License compliance
  - Dependency changes
- **Triggers**: Only on pull requests
- **Cost**: Free

```yaml
- name: Dependency Review
  uses: actions/dependency-review-action@v4
```

#### 3. **pip-audit (Vulnerability Scan)** 🔬
- **Type**: GitHub Actions Workflow
- **File**: `.github/workflows/security.yml`
- **Tool**: pip-audit (Python dependency scanner)
- **How it works**: Scans Python dependencies for known CVEs
- **Coverage**:
  - PyPI vulnerability database
  - OSV (Open Source Vulnerabilities)
- **Cost**: Free (uses GitHub Actions minutes)

```yaml
- name: Run pip-audit
  run: |
    uv run pip-audit
```

#### 4. **CodeQL Analysis (Advanced)** 🔍
- **Type**: GitHub Actions Workflow (when using advanced setup)
- **File**: `.github/workflows/codeql.yml` or `.github/workflows/security.yml`
- **Action**: `github/codeql-action`
- **How it works**: Deep static analysis of code
- **Coverage**:
  - Security vulnerabilities
  - Code quality issues
  - Custom queries
- **Cost**: Free for public repos

```yaml
- name: Initialize CodeQL
  uses: github/codeql-action/init@v3
  with:
    languages: python

- name: Perform CodeQL Analysis
  uses: github/codeql-action/analyze@v3
```

## 🔄 Comparison Matrix

| Feature | Type | Configuration | Triggers | Cost | Alerts Location |
|---------|------|---------------|----------|------|-----------------|
| **Secret Scanning** | Native | Auto-enabled | Every commit | Free (public) | Security tab |
| **Dependabot Alerts** | Native | Auto-enabled | Dependency updates | Free | Security tab |
| **Dependabot Updates** | Native | Enable in settings | Vulnerability found | Free | Pull requests |
| **CodeQL (Default)** | Native | Auto-enabled | Push/PR | Free (public) | Security tab |
| **CodeQL (Advanced)** | Actions | Workflow file | Push/PR | Free (public) | Security tab |
| **Bandit** | Actions | Workflow file | Push/PR | Free | Workflow logs |
| **Dependency Review** | Actions | Workflow file | PR only | Free | Workflow logs |
| **pip-audit** | Actions | Workflow file | Push/PR | Free | Workflow logs |

## 📋 Current Project Configuration

### Native Features Enabled
✅ **Secret Scanning**: Enabled  
✅ **Dependabot Alerts**: Enabled  
✅ **Dependabot Security Updates**: Enabled  
✅ **CodeQL (Default Setup)**: Enabled  

### GitHub Actions Workflows

#### `.github/workflows/security.yml`
```yaml
name: Security Checks

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  bandit:
    name: Bandit Security Linter
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install uv
        uses: astral-sh/setup-uv@v5
      - name: Run Bandit
        run: |
          uv run bandit -r src/ -c pyproject.toml

  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
    steps:
      - uses: actions/checkout@v4
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: python
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3

  dependency-review:
    name: Dependency Review
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - name: Dependency Review
        uses: actions/dependency-review-action@v4

  vulnerability-scan:
    name: Vulnerability Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install uv
        uses: astral-sh/setup-uv@v5
      - name: Run pip-audit
        run: |
          uv run pip-audit

  secret-scanning:
    name: Secret Scanning
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: TruffleHog OSS
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD
```

## 🎯 Best Practices

### For Native Features

1. **Enable all native security features**:
   - Go to Settings → Security → Code security and analysis
   - Enable Secret scanning, Dependabot alerts, and Code scanning

2. **Review alerts regularly**:
   - Check Security tab weekly
   - Prioritize high-severity alerts
   - Create issues for tracking

3. **Configure Dependabot**:
   - Create `.github/dependabot.yml` for custom configuration
   - Set update frequency and reviewers

### For GitHub Actions Workflows

1. **Use specific versions**:
   ```yaml
   uses: actions/checkout@v4  # Good - specific version
   uses: actions/checkout@main  # Bad - unpredictable
   ```

2. **Set appropriate permissions**:
   ```yaml
   permissions:
     contents: read
     security-events: write
   ```

3. **Cache dependencies**:
   ```yaml
   - name: Cache uv
     uses: actions/cache@v3
     with:
       path: ~/.cache/uv
       key: ${{ runner.os }}-uv-${{ hashFiles('**/pyproject.toml') }}
   ```

4. **Fail on security issues**:
   ```yaml
   - name: Run Bandit
     run: |
       uv run bandit -r src/ -c pyproject.toml
       # Exit code 1 if issues found
   ```

## 🔧 Troubleshooting

### Native Features Not Working

**Issue**: Secret scanning not detecting secrets
- **Solution**: Check if repository is public or has GitHub Advanced Security
- **Verify**: Settings → Security → Code security and analysis

**Issue**: Dependabot not creating PRs
- **Solution**: 
  1. Check if Dependabot security updates are enabled
  2. Verify `.github/dependabot.yml` configuration
  3. Check for existing PRs (Dependabot limits concurrent PRs)

**Issue**: CodeQL not running
- **Solution**:
  1. Check if default setup is enabled
  2. Verify language support (Python is supported)
  3. Check workflow permissions

### GitHub Actions Issues

**Issue**: Bandit workflow failing
- **Solution**: Run locally first: `uv run bandit -r src/ -c pyproject.toml`
- **Check**: pyproject.toml configuration

**Issue**: CodeQL analysis timing out
- **Solution**: 
  1. Increase timeout: `timeout-minutes: 30`
  2. Reduce code size or complexity
  3. Use matrix builds for large codebases

**Issue**: pip-audit finding vulnerabilities
- **Solution**: Update dependencies: `uv sync --upgrade`
- **Check**: Review vulnerability details and update specific packages

## 📚 Additional Resources

### GitHub Documentation
- [About secret scanning](https://docs.github.com/en/code-security/secret-scanning/about-secret-scanning)
- [About Dependabot](https://docs.github.com/en/code-security/dependabot)
- [About code scanning](https://docs.github.com/en/code-security/code-scanning)
- [About dependency review](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/about-dependency-review)

### Tool Documentation
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [pip-audit Documentation](https://pypi.org/project/pip-audit/)
- [CodeQL Documentation](https://codeql.github.com/docs/)

## 🆘 Getting Help

If you encounter issues with security scanning:

1. **Check GitHub Status**: [GitHub Status](https://www.githubstatus.com/)
2. **Review Workflow Logs**: Actions tab → Select workflow run
3. **Test Locally**: Run security tools locally before pushing
4. **Contact Support**: GitHub Support or repository maintainers

---

*Last updated: 2026-01-31*
