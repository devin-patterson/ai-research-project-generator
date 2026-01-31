"""
FastAPI Application Entry Point

Production-grade API for AI Research Project Generation.
"""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from ..api.routes import router
from ..core.config import get_settings
from ..core.exceptions import ResearchGenerationError
from ..services.research_service import ResearchService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events for resource management.
    """
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Startup: Initialize resources
    global _research_service
    from ..api import routes

    routes._research_service = ResearchService(settings)
    await routes._research_service.startup()

    logger.info("Application startup complete")

    yield

    # Shutdown: Cleanup resources
    logger.info("Shutting down application...")
    if routes._research_service:
        await routes._research_service.shutdown()

    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """
    Application factory.

    Creates and configures the FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="""
## AI Research Project Generator API

Generate comprehensive, AI-enhanced research projects with:

- 🤖 **Local LLM Integration** - Topic analysis, research questions, methodology recommendations
- 📚 **Academic Search** - Real papers from Semantic Scholar, OpenAlex, CrossRef
- ✅ **Quality Validation** - PRISMA compliance, methodological rigor assessment
- 📊 **Multiple Research Types** - Systematic reviews, meta-analyses, qualitative studies, and more

### Quick Start

1. **Health Check**: `GET /api/v1/health`
2. **Generate Project**: `POST /api/v1/research`
3. **List Models**: `GET /api/v1/models`

### Example Request

```json
{
    "topic": "Impact of remote work on employee productivity",
    "research_question": "How does remote work affect productivity?",
    "research_type": "systematic_review",
    "discipline": "psychology"
}
```
        """,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(ResearchGenerationError)
    async def research_error_handler(request: Request, exc: ResearchGenerationError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": exc.message,
                "error_code": exc.error_code,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal error occurred",
                "timestamp": datetime.now().isoformat(),
            },
        )

    # Include routers
    app.include_router(router, prefix=settings.api_prefix, tags=["Research"])

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": f"{settings.api_prefix}/health",
        }

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
