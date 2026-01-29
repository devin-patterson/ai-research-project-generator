# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| 1.x.x   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email the maintainers directly or use GitHub's private vulnerability reporting feature
3. Include as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Assessment**: We will assess the vulnerability and determine its severity
- **Updates**: We will keep you informed of our progress
- **Resolution**: We aim to resolve critical vulnerabilities within 7 days
- **Credit**: We will credit you in the release notes (unless you prefer anonymity)

## Security Best Practices

When using this project:

### API Keys and Secrets

- **Never commit API keys** to version control
- Use environment variables or `.env` files (which are gitignored)
- Rotate API keys regularly
- Use the minimum required permissions

### LLM Security

- **Local LLM (Ollama)**: Data stays on your machine
- **External APIs**: Be aware that prompts are sent to external services
- **Sensitive Data**: Do not include sensitive information in research prompts

### Deployment

- Use HTTPS in production
- Implement rate limiting
- Use authentication for public deployments
- Keep dependencies updated

## Security Features

This project includes several security measures:

### Automated Security Scanning

- **CodeQL**: Static analysis for security vulnerabilities
- **Dependabot**: Automatic dependency updates
- **pip-audit**: Python dependency vulnerability scanning
- **Bandit**: Python security linter
- **TruffleHog**: Secret scanning

### Input Validation

- Pydantic schemas validate all API inputs
- Maximum length limits on text fields
- Type checking throughout

### Dependency Management

- Lock files ensure reproducible builds
- Regular dependency audits
- Automated security updates via Dependabot

## Known Security Considerations

### External API Calls

This application makes calls to external services:
- Ollama (local or remote)
- OpenAI API (if configured)
- Semantic Scholar API
- OpenAlex API
- CrossRef API
- arXiv API

Ensure you trust these services and understand their data handling policies.

### Rate Limiting

The application includes rate limiting for external APIs but does not implement rate limiting on its own endpoints. For production deployments, consider adding:
- API rate limiting
- Request throttling
- Authentication/authorization

## Updates

This security policy may be updated periodically. Check back for the latest version.

Last updated: January 2026
