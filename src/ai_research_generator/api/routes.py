"""
API Routes for Research Project Generator

Defines all REST API endpoints with proper request/response schemas.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from ..core.config import Settings, get_settings
from ..core.exceptions import ResearchGenerationError, LLMConnectionError
from ..models.schemas.research import (
    ResearchRequest,
    ResearchResponse,
    HealthResponse,
    ErrorResponse,
)
from ..services.research_service import ResearchService
from ..templates import get_template_manager
from datetime import datetime

router = APIRouter()

# Dependency for research service
_research_service: ResearchService | None = None


async def get_research_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ResearchService:
    """Dependency injection for research service"""
    global _research_service
    if _research_service is None:
        _research_service = ResearchService(settings)
        await _research_service.startup()
    return _research_service


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API and its dependencies",
)
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> HealthResponse:
    """
    Health check endpoint.

    Returns the status of the API and its dependencies including:
    - LLM availability
    - Academic search availability
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        llm_available=service.llm_available,
        llm_model=settings.llm_model if service.llm_available else None,
        academic_search_available=service.academic_search_available,
        timestamp=datetime.now(),
    )


@router.post(
    "/research",
    response_model=ResearchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate Research Project",
    description="Generate a comprehensive AI-enhanced research project",
    responses={
        201: {"description": "Research project generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Generation failed"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def generate_research_project(
    request: ResearchRequest,
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchResponse:
    """
    Generate a comprehensive research project.

    This endpoint accepts a research topic and configuration, then generates:
    - AI-powered topic analysis
    - Research questions
    - Methodology recommendations
    - Literature search strategy
    - Relevant academic papers
    - Literature synthesis
    - Project validation report

    **Example Request:**
    ```json
    {
        "topic": "Impact of remote work on employee productivity",
        "research_question": "How does remote work affect productivity?",
        "research_type": "systematic_review",
        "academic_level": "graduate",
        "discipline": "psychology",
        "search_papers": true,
        "paper_limit": 20,
        "use_llm": true
    }
    ```
    """
    try:
        logger.info(f"Received research request: {request.topic[:50]}...")
        response = await service.generate_project(request)
        logger.info(f"Research project generated: {response.request_id}")
        return response

    except LLMConnectionError as e:
        logger.error(f"LLM connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service unavailable: {str(e)}",
        )
    except ResearchGenerationError as e:
        logger.error(f"Research generation error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.get(
    "/models",
    summary="List Available Models",
    description="List recommended LLM models for research generation",
)
async def list_models() -> dict:
    """
    List recommended LLM models for research generation.

    Returns models categorized by:
    - Best quality (24GB+ VRAM)
    - Balanced (8-16GB VRAM)
    - Lightweight (4-8GB VRAM)
    - Reasoning-focused
    """
    return {
        "recommended_models": {
            "best_quality": [
                {"name": "llama3.1:70b", "vram": "48GB+", "context": "128K"},
                {"name": "qwen2.5:32b", "vram": "24GB+", "context": "128K"},
            ],
            "balanced": [
                {"name": "llama3.1:8b", "vram": "8GB+", "context": "128K"},
                {"name": "qwen2.5:14b", "vram": "12GB+", "context": "128K"},
            ],
            "lightweight": [
                {"name": "llama3.2:3b", "vram": "4GB+", "context": "128K"},
                {"name": "phi4:3.8b", "vram": "4GB+", "context": "128K"},
            ],
            "reasoning": [
                {"name": "deepseek-r1:7b", "vram": "8GB+", "context": "64K"},
            ],
        }
    }


@router.get(
    "/research-types",
    summary="List Research Types",
    description="List supported research methodologies",
)
async def list_research_types() -> dict:
    """List all supported research types with descriptions"""
    return {
        "research_types": [
            {
                "value": "systematic_review",
                "name": "Systematic Review",
                "description": "PRISMA-compliant systematic literature review",
            },
            {
                "value": "scoping_review",
                "name": "Scoping Review",
                "description": "Broad literature mapping and gap identification",
            },
            {
                "value": "meta_analysis",
                "name": "Meta-Analysis",
                "description": "Statistical synthesis of multiple studies",
            },
            {
                "value": "qualitative_study",
                "name": "Qualitative Study",
                "description": "In-depth exploration of phenomena",
            },
            {
                "value": "quantitative_study",
                "name": "Quantitative Study",
                "description": "Statistical analysis of measurable data",
            },
            {
                "value": "mixed_methods",
                "name": "Mixed Methods",
                "description": "Combined qualitative and quantitative approaches",
            },
            {
                "value": "case_study",
                "name": "Case Study",
                "description": "In-depth analysis of specific cases",
            },
            {
                "value": "experimental",
                "name": "Experimental",
                "description": "Controlled experimental research",
            },
            {
                "value": "literature_review",
                "name": "Literature Review",
                "description": "Narrative review of existing literature",
            },
        ]
    }


# =============================================================================
# Template Endpoints
# =============================================================================


@router.get(
    "/templates",
    summary="List Research Templates",
    description="List all available research templates with their parameters",
)
async def list_templates() -> dict:
    """
    List all available research templates.

    Templates provide pre-configured research setups for common use cases
    with specialized parameters for more valuable and robust research output.
    """
    manager = get_template_manager()
    return {
        "templates": manager.list_templates(),
        "categories": manager.get_categories(),
        "tags": manager.get_tags(),
    }


@router.get(
    "/templates/{template_id}",
    summary="Get Template Details",
    description="Get detailed information about a specific template",
)
async def get_template(template_id: str) -> dict:
    """
    Get detailed information about a specific template.

    Returns the template metadata and all available parameters.
    """
    manager = get_template_manager()
    template = manager.get(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    return {"template": template.to_dict(), "defaults": template.get_defaults()}


@router.post(
    "/templates/{template_id}/research",
    response_model=ResearchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate Research from Template",
    description="Generate a research project using a template with custom parameters",
    responses={
        201: {"description": "Research project generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        404: {"model": ErrorResponse, "description": "Template not found"},
        500: {"model": ErrorResponse, "description": "Generation failed"},
    },
)
async def generate_from_template(
    template_id: str,
    params: dict,
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchResponse:
    """
    Generate a research project using a template.

    **Example Request for Investment Template:**
    ```json
    {
        "investment_goal": "long_term_growth",
        "investment_horizon": 15,
        "risk_tolerance": "moderate",
        "asset_classes": ["stocks", "bonds", "etfs", "real_estate"],
        "geographic_focus": ["us", "international_developed"],
        "market_outlook": "moderately_bullish",
        "inflation_expectation": "moderate",
        "dividend_preference": "balanced",
        "include_risk_metrics": true,
        "paper_limit": 30
    }
    ```
    """
    manager = get_template_manager()

    # Validate template exists
    template = manager.get(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )

    # Validate parameters
    is_valid, errors = manager.validate_params(template_id, params)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameters: {'; '.join(errors)}",
        )

    try:
        # Create research request from template
        request_dict = manager.create_research_request(template_id, params)
        request = ResearchRequest(**request_dict)

        logger.info(f"Generating research from template '{template_id}': {request.topic[:50]}...")
        response = await service.generate_project(request)
        logger.info(f"Template research generated: {response.request_id}")
        return response

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except LLMConnectionError as e:
        logger.error(f"LLM connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service unavailable: {str(e)}",
        )
    except ResearchGenerationError as e:
        logger.error(f"Research generation error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.post(
    "/templates/{template_id}/validate",
    summary="Validate Template Parameters",
    description="Validate parameters for a template without generating research",
)
async def validate_template_params(template_id: str, params: dict) -> dict:
    """
    Validate parameters for a template.

    Returns validation status and any errors found.
    """
    manager = get_template_manager()

    template = manager.get(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )

    is_valid, errors = manager.validate_params(template_id, params)
    return {
        "valid": is_valid,
        "errors": errors,
        "merged_params": {**template.get_defaults(), **params},
    }


@router.post(
    "/templates/{template_id}/preview",
    summary="Preview Research Request",
    description="Preview the research request that would be generated from template parameters",
)
async def preview_template_request(template_id: str, params: dict) -> dict:
    """
    Preview the research request that would be generated.

    Returns the topic, research question, and additional context
    without actually generating the research.
    """
    manager = get_template_manager()

    template = manager.get(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )

    try:
        request_dict = manager.create_research_request(template_id, params)
        return {
            "preview": {
                "topic": request_dict["topic"],
                "research_question": request_dict["research_question"],
                "research_type": request_dict["research_type"],
                "discipline": request_dict["discipline"],
                "academic_level": request_dict["academic_level"],
                "additional_context": request_dict["additional_context"],
                "paper_limit": request_dict["paper_limit"],
            },
            "search_keywords": template.get_search_keywords(params),
            "recommended_sources": template.get_recommended_sources(),
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
