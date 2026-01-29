# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI/CD workflows
- Security scanning (CodeQL, Bandit, pip-audit, TruffleHog)
- Dependabot configuration for automated updates
- Issue and PR templates
- CONTRIBUTING.md guide
- SECURITY.md policy

## [2.0.0] - 2026-01-29

### Added
- **FastAPI REST API** with production-grade architecture
- **Pydantic schemas** for request/response validation
- **Service layer pattern** for business logic encapsulation
- **Dependency injection** using FastAPI's Depends system
- **Application factory** with async lifespan management
- **Custom exception hierarchy** for better error handling
- **OpenAPI documentation** auto-generated at `/docs`
- **Pydantic Settings** for environment-based configuration
- **Health check endpoint** at `/api/v1/health`
- **Research types endpoint** at `/api/v1/research-types`
- **Models endpoint** at `/api/v1/models`

### Changed
- Refactored from CLI-only to API-first architecture
- Moved business logic to service layer
- Updated configuration to use Pydantic Settings
- Improved error handling with specific exception types

### Fixed
- ValidationIssue attribute access in validation report
- OpenAlex search parameter handling

## [1.0.0] - 2026-01-29

### Added
- **Local LLM integration** via Ollama
- **OpenAI-compatible API** support
- **Academic search APIs**:
  - Semantic Scholar
  - OpenAlex
  - CrossRef
  - arXiv
- **Research LLM Assistant** with research-specific prompts
- **Unified academic search** interface with deduplication
- **AI Research Engine** combining all components
- **Rule-based project generation**
- **Subject analysis** module
- **Validation engine** for quality checks
- **CLI interface** for command-line usage
- **Markdown export** for research projects

### Technical
- Python 3.10+ support
- `uv` package manager
- `httpx` for async HTTP
- `loguru` for logging
- Type hints throughout

## [0.1.0] - 2026-01-29

### Added
- Initial project structure
- Basic research project generator
- Subject analyzer
- Validation engine

[Unreleased]: https://github.com/depatter/ai-research-project-generator/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/depatter/ai-research-project-generator/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/depatter/ai-research-project-generator/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/depatter/ai-research-project-generator/releases/tag/v0.1.0
