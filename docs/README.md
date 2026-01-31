# AI Research Project Generator Documentation

Welcome to the comprehensive documentation for the AI Research Project Generator.

## 📚 Documentation Structure

### Overview
- [**Quick Start**](../README.md) - Installation and basic usage
- [**Architecture**](ARCHITECTURE.md) - System design and patterns
- [**Development**](development/CONTRIBUTING.md) - Setup and contribution guide
- [**Deployment**](deployment/SECURITY.md) - Production deployment guide
- [**Changelog**](CHANGELOG.md) - Version history and updates

### API Documentation
- [**API Reference**](api/) - REST API endpoints and schemas
- [**AI Components**](ai/) - AI agents, workflows, and optimization

### AI Enablement
- [**AI Architecture**](ai/AI_ENABLEMENT.md) - AI components overview
- [**PydanticAI Agents**](ai/PYDANTIC_AI.md) - Type-safe LLM agents
- [**LangGraph Workflows**](ai/LANGGRAPH.md) - Stateful workflow orchestration
- [**DSPy Optimization**](ai/DSPY.md) - Prompt optimization framework

## 🚀 Quick Links

### For Users
- [Installation Guide](../README.md#installation)
- [Basic Usage](../README.md#usage)
- [API Documentation](api/)

### For Developers
- [Development Setup](development/CONTRIBUTING.md)
- [Architecture Overview](ARCHITECTURE.md)
- [AI Components Guide](ai/AI_ENABLEMENT.md)

### For Operations
- [Deployment Guide](deployment/SECURITY.md)
- [Security Best Practices](deployment/SECURITY.md)
- [Monitoring and Observability](deployment/SECURITY.md)

## 🏗️ Architecture Overview

The AI Research Project Generator follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application Layer                │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   API Routes  │  │   Models     │  │   Services   │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         ▼                 ▼                 ▼                │
│  ┌──────────────────────────────────────────────────┐       │
│  │              Business Logic Layer                │       │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │       │
│  │  │    Agents    │  │  Workflows   │  │ Optimization │ │       │
│  │  └──────────────┘  └──────────────┘  └───────────────┘ │       │
│  └──────────────────────────────────────────────────┘       │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────┐       │
│  │                Legacy Compatibility Layer        │       │
│  │  (Academic Search, LLM Provider, etc.)           │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## 🤖 AI Enablement Status

### ✅ Implemented AI Components

| Component | Framework | Status | Description |
|------------|-----------|---------|-------------|
| **PydanticAI Agents** | PydanticAI | ✅ Active | Type-safe structured LLM output |
| **LangGraph Workflows** | LangGraph | ✅ Active | Stateful research workflow orchestration |
| **DSPy Optimization** | DSPy | ✅ Active | Offline prompt optimization |
| **DeepEval Testing** | DeepEval | ✅ Active | LLM output evaluation |

### 🔧 Integration Status

- **FastAPI Integration**: AI components are integrated into the REST API
- **Configuration Management**: Unified configuration for all AI services
- **Error Handling**: Comprehensive error handling and retry policies
- **Observability**: Logging and monitoring for AI operations

## 📖 Getting Started

1. **For API Users**: See the [API Documentation](api/)
2. **For Developers**: Start with the [Development Guide](development/CONTRIBUTING.md)
3. **For AI Integration**: Read the [AI Enablement Guide](ai/AI_ENABLEMENT.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](development/CONTRIBUTING.md) for details on how to get started.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
