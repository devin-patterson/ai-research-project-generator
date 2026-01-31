"""
PydanticAI agents for structured LLM output in research workflows.

This module defines type-safe agents using PydanticAI that guarantee
structured, validated output from LLM calls.
"""

from dataclasses import dataclass
from typing import Optional, Any

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent


# =============================================================================
# Dependencies
# =============================================================================


@dataclass
class ResearchDependencies:
    """Dependencies injected into research agents.

    This dataclass provides type-safe access to external services
    and configuration needed by the agents.
    """

    http_client: httpx.AsyncClient
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    api_timeout: float = 120.0


# =============================================================================
# Output Schemas (Pydantic Models)
# =============================================================================


class TopicAnalysis(BaseModel):
    """Structured output for topic analysis."""

    topic: str = Field(description="The research topic being analyzed")
    key_concepts: list[str] = Field(
        description="Key concepts and terms identified in the topic", min_length=1, max_length=20
    )
    research_scope: str = Field(description="The scope of the research (narrow, moderate, broad)")
    complexity_level: str = Field(description="Complexity level (basic, intermediate, advanced)")
    suggested_subtopics: list[str] = Field(
        description="Suggested subtopics for deeper investigation", default_factory=list
    )
    potential_challenges: list[str] = Field(
        description="Potential challenges in researching this topic", default_factory=list
    )
    interdisciplinary_connections: list[str] = Field(
        description="Related disciplines that may inform this research", default_factory=list
    )


class PaperSynthesis(BaseModel):
    """Structured output for paper synthesis."""

    papers_analyzed: int = Field(description="Number of papers analyzed", ge=0)
    main_themes: list[str] = Field(description="Main themes identified across papers")
    research_gaps: list[str] = Field(description="Gaps in the current research identified")
    methodological_trends: list[str] = Field(
        description="Common methodological approaches observed"
    )
    key_findings: list[str] = Field(description="Key findings synthesized from the literature")
    synthesis_narrative: str = Field(description="Narrative synthesis of the literature")
    recommended_focus_areas: list[str] = Field(
        description="Recommended areas for further research", default_factory=list
    )


class MethodologyRecommendation(BaseModel):
    """Structured output for methodology recommendations."""

    primary_methodology: str = Field(description="Primary recommended research methodology")
    rationale: str = Field(description="Rationale for the methodology recommendation")
    data_collection_methods: list[str] = Field(description="Recommended data collection methods")
    analysis_techniques: list[str] = Field(description="Recommended analysis techniques")
    quality_criteria: list[str] = Field(description="Quality criteria to ensure rigor")
    potential_limitations: list[str] = Field(description="Potential limitations of this approach")
    alternative_approaches: list[str] = Field(
        description="Alternative methodological approaches", default_factory=list
    )


class ResearchProjectOutput(BaseModel):
    """Complete structured output for a research project."""

    title: str = Field(description="Suggested title for the research project")
    abstract: str = Field(description="Brief abstract of the research project")
    topic_analysis: TopicAnalysis
    literature_synthesis: Optional[PaperSynthesis] = None
    methodology: MethodologyRecommendation
    research_questions: list[str] = Field(description="Refined research questions")
    hypotheses: list[str] = Field(
        description="Research hypotheses if applicable", default_factory=list
    )
    expected_contributions: list[str] = Field(description="Expected contributions to the field")
    timeline_estimate: str = Field(description="Estimated timeline for the research")
    quality_score: float = Field(description="Quality score of the generated project", ge=0, le=100)


# =============================================================================
# PydanticAI Agents
# =============================================================================

# Topic Analysis Agent
# Note: Agent initialization is deferred to avoid requiring OLLAMA_BASE_URL at import time
# Use create_topic_analysis_agent() factory function instead


def create_topic_analysis_agent(model: str = "ollama:llama3.1:8b") -> Agent:
    """Factory function to create topic analysis agent."""
    return Agent(
        model,
        deps_type=ResearchDependencies,
        output_type=TopicAnalysis,
        system_prompt="""You are an expert research analyst specializing in academic topic analysis.

Your task is to analyze research topics and extract:
1. Key concepts and terminology
2. Research scope and complexity
3. Potential subtopics for investigation
4. Challenges and interdisciplinary connections

Provide thorough, academically rigorous analysis that helps researchers 
understand the landscape of their chosen topic.""",
    )


def create_paper_synthesis_agent(model: str = "ollama:llama3.1:8b") -> Agent:
    """Factory function to create paper synthesis agent."""
    return Agent(
        model,
        deps_type=ResearchDependencies,
        output_type=PaperSynthesis,
        system_prompt="""You are an expert at synthesizing academic literature.

Your task is to analyze collections of academic papers and:
1. Identify main themes and patterns
2. Find gaps in the current research
3. Note methodological trends
4. Synthesize key findings into a coherent narrative

Provide rigorous, balanced synthesis that accurately represents the literature.""",
    )


def create_methodology_agent(model: str = "ollama:llama3.1:8b") -> Agent:
    """Factory function to create methodology recommendation agent."""
    return Agent(
        model,
        deps_type=ResearchDependencies,
        output_type=MethodologyRecommendation,
        system_prompt="""You are an expert research methodologist.

Your task is to recommend appropriate research methodologies based on:
1. The research topic and questions
2. The discipline and academic level
3. Available resources and constraints
4. Best practices in the field

Provide practical, well-justified methodology recommendations.""",
    )


def create_research_project_agent(model: str = "ollama:llama3.1:8b") -> Agent:
    """Factory function to create complete research project agent."""
    return Agent(
        model,
        deps_type=ResearchDependencies,
        output_type=ResearchProjectOutput,
        system_prompt="""You are an expert research project designer.

Your task is to create comprehensive research project plans that include:
1. Clear topic analysis and scope definition
2. Literature synthesis (if papers provided)
3. Appropriate methodology recommendations
4. Refined research questions and hypotheses
5. Expected contributions and timeline

Create academically rigorous, feasible research projects.""",
    )


# =============================================================================
# Agent Execution Functions
# =============================================================================


async def analyze_topic_with_agent(
    topic: str,
    research_question: str,
    deps: ResearchDependencies,
    model: str = "ollama:llama3.1:8b",
) -> TopicAnalysis:
    """
    Analyze a research topic using the PydanticAI agent.

    Args:
        topic: The research topic to analyze
        research_question: The main research question
        deps: Dependencies for the agent
        model: LLM model to use

    Returns:
        Structured TopicAnalysis output
    """
    agent = create_topic_analysis_agent(model)

    prompt = f"""Analyze the following research topic:

Topic: {topic}
Research Question: {research_question}

Provide a comprehensive analysis including key concepts, scope, 
complexity, subtopics, challenges, and interdisciplinary connections."""

    result = await agent.run(prompt, deps=deps)
    return result.data


async def synthesize_papers_with_agent(
    papers: list[dict[str, Any]],
    topic: str,
    deps: ResearchDependencies,
    model: str = "ollama:llama3.1:8b",
) -> PaperSynthesis:
    """
    Synthesize academic papers using the PydanticAI agent.

    Args:
        papers: List of paper metadata dictionaries
        topic: The research topic
        deps: Dependencies for the agent
        model: LLM model to use

    Returns:
        Structured PaperSynthesis output
    """
    agent = create_paper_synthesis_agent(model)

    # Format papers for the prompt
    paper_summaries = []
    for i, paper in enumerate(papers[:20], 1):  # Limit to 20 papers
        title = paper.get("title", "Unknown")
        abstract = paper.get("abstract", "No abstract available")[:500]
        year = paper.get("year", "N/A")
        paper_summaries.append(f"{i}. {title} ({year})\n   {abstract}")

    papers_text = "\n\n".join(paper_summaries)

    prompt = f"""Synthesize the following academic papers on the topic: {topic}

Papers:
{papers_text}

Provide a comprehensive synthesis including main themes, research gaps,
methodological trends, key findings, and recommended focus areas."""

    result = await agent.run(prompt, deps=deps)
    return result.data


async def recommend_methodology_with_agent(
    topic: str,
    research_type: str,
    discipline: str,
    academic_level: str,
    deps: ResearchDependencies,
    model: str = "ollama:llama3.1:8b",
) -> MethodologyRecommendation:
    """
    Recommend research methodology using the PydanticAI agent.

    Args:
        topic: The research topic
        research_type: Type of research (e.g., systematic_review)
        discipline: Academic discipline
        academic_level: Academic level (undergraduate, graduate, doctoral)
        deps: Dependencies for the agent
        model: LLM model to use

    Returns:
        Structured MethodologyRecommendation output
    """
    agent = create_methodology_agent(model)

    prompt = f"""Recommend a research methodology for the following:

Topic: {topic}
Research Type: {research_type}
Discipline: {discipline}
Academic Level: {academic_level}

Provide detailed methodology recommendations including data collection,
analysis techniques, quality criteria, and potential limitations."""

    result = await agent.run(prompt, deps=deps)
    return result.data


async def generate_research_project_with_agent(
    topic: str,
    research_question: str,
    research_type: str,
    discipline: str,
    academic_level: str,
    papers: Optional[list[dict[str, Any]]] = None,
    deps: Optional[ResearchDependencies] = None,
    model: str = "ollama:llama3.1:8b",
) -> ResearchProjectOutput:
    """
    Generate a complete research project using the PydanticAI agent.

    Args:
        topic: The research topic
        research_question: The main research question
        research_type: Type of research
        discipline: Academic discipline
        academic_level: Academic level
        papers: Optional list of papers to include
        deps: Dependencies for the agent
        model: LLM model to use

    Returns:
        Structured ResearchProjectOutput
    """
    if deps is None:
        async with httpx.AsyncClient() as client:
            deps = ResearchDependencies(http_client=client)
            return await _run_research_project_agent(
                topic,
                research_question,
                research_type,
                discipline,
                academic_level,
                papers,
                deps,
                model,
            )
    else:
        return await _run_research_project_agent(
            topic, research_question, research_type, discipline, academic_level, papers, deps, model
        )


async def _run_research_project_agent(
    topic: str,
    research_question: str,
    research_type: str,
    discipline: str,
    academic_level: str,
    papers: Optional[list[dict[str, Any]]],
    deps: ResearchDependencies,
    model: str = "ollama:llama3.1:8b",
) -> ResearchProjectOutput:
    """Internal function to run the research project agent."""
    agent = create_research_project_agent(model)

    papers_section = ""
    if papers:
        paper_list = "\n".join(
            [f"- {p.get('title', 'Unknown')} ({p.get('year', 'N/A')})" for p in papers[:10]]
        )
        papers_section = f"\n\nRelevant Papers Found:\n{paper_list}"

    prompt = f"""Create a comprehensive research project plan:

Topic: {topic}
Research Question: {research_question}
Research Type: {research_type}
Discipline: {discipline}
Academic Level: {academic_level}
{papers_section}

Generate a complete research project including title, abstract,
topic analysis, methodology, research questions, hypotheses,
expected contributions, and timeline."""

    result = await agent.run(prompt, deps=deps)
    return result.data
