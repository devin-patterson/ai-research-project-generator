# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| 1.x.x   | :x:                |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email your findings to the maintainers (see CODEOWNERS)
3. Or use GitHub's private vulnerability reporting feature

### What to Include

Please include the following information in your report:

- Type of vulnerability (e.g., SQL injection, XSS, authentication bypass)
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 30 days (depending on complexity)

### What to Expect

1. We will acknowledge receipt of your vulnerability report
2. We will investigate and validate the reported vulnerability
3. We will work on a fix and coordinate disclosure timing with you
4. We will credit you in the security advisory (unless you prefer anonymity)

## Security Best Practices for Users

### API Keys and Secrets

- **Never commit API keys** to version control
- Use environment variables or `.env` files (which are gitignored)
- Rotate API keys regularly
- Use the minimum required permissions for API keys

### LLM Security

- **Local LLM (Ollama)**: Recommended for sensitive data - data never leaves your machine
- **Cloud LLM APIs**: Be aware that prompts may be logged by providers
- **Input Sanitization**: The application sanitizes user inputs before sending to LLMs

### Network Security

- Run the API server behind a reverse proxy in production
- Use HTTPS in production environments
- Configure CORS appropriately for your deployment

### Dependencies

- We use Dependabot for automated dependency updates
- Security scanning runs on every PR via GitHub Actions
- We use `pip-audit` and `bandit` for vulnerability scanning

## Security Features

### Built-in Security Measures

1. **Input Validation**: Pydantic schemas validate all API inputs
2. **Rate Limiting**: Academic search APIs have built-in rate limiting
3. **No Data Persistence**: By default, no user data is stored
4. **Dependency Scanning**: Automated security scanning in CI/CD

### Security-Related Configuration

```bash
# Recommended .env configuration
LLM_PROVIDER=ollama          # Use local LLM for sensitive data
LLM_BASE_URL=http://localhost:11434
# API keys should be set as environment variables, not in files
```

## Vulnerability Disclosure Policy

We follow a coordinated disclosure policy:

1. Reporter submits vulnerability privately
2. We acknowledge and investigate
3. We develop and test a fix
4. We release the fix and publish a security advisory
5. Reporter may publish their findings after the fix is released

## Past Security Advisories

No security advisories have been published yet.

---

Thank you for helping keep AI Research Project Generator and its users safe!
