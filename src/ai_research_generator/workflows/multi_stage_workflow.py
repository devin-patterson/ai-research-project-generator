"""
Multi-Stage Research Workflow

Implements advanced multi-stage research workflows using LangGraph.
Supports configurable research stages, parallel execution, and 
comprehensive research report generation.

Stages:
1. Discovery - Topic analysis and scope definition
2. Collection - Multi-source data gathering (academic, web, news)
3. Analysis - Deep analysis and synthesis
4. Verification - Fact checking and source validation
5. Synthesis - Knowledge integration and gap identification
6. Report - Comprehensive report generation with citations
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from operator import add

from loguru import logger
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field


class ResearchStage(str, Enum):
    """Research workflow stages"""
    DISCOVERY = "discovery"
    COLLECTION = "collection"
    ANALYSIS = "analysis"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    REPORT = "report"


class StageStatus(str, Enum):
    """Status of a workflow stage"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageResult(BaseModel):
    """Result from a workflow stage"""
    stage: ResearchStage
    status: StageStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class MultiStageState(TypedDict, total=False):
    """State schema for multi-stage research workflow"""
    
    # Request identification
    request_id: str
    created_at: str
    
    # Input parameters
    topic: str
    research_question: str
    research_type: str
    discipline: str
    academic_level: str
    additional_context: Optional[str]
    
    # Configuration
    stages_config: Dict[str, Any]
    use_llm: bool
    use_web_search: bool
    use_academic_search: bool
    use_google_scholar: bool
    use_fact_verification: bool
    citation_style: str
    max_sources: int
    
    # Stage tracking
    current_stage: str
    stage_results: Dict[str, Dict[str, Any]]
    errors: Annotated[List[str], add]
    
    # Discovery stage outputs
    topic_analysis: Dict[str, Any]
    key_concepts: List[str]
    research_scope: Dict[str, Any]
    search_strategy: Dict[str, Any]
    
    # Collection stage outputs
    academic_papers: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    google_scholar_papers: List[Dict[str, Any]]
    all_sources: List[Dict[str, Any]]
    
    # Analysis stage outputs
    source_analysis: Dict[str, Any]
    theme_analysis: Dict[str, Any]
    methodology_analysis: Dict[str, Any]
    
    # Verification stage outputs
    verified_claims: List[Dict[str, Any]]
    source_credibility: Dict[str, Any]
    
    # Synthesis stage outputs
    synthesized_knowledge: Dict[str, Any]
    knowledge_gaps: List[str]
    recommendations: List[str]
    
    # Report stage outputs
    citations: List[Dict[str, Any]]
    bibliography: str
    executive_summary: str
    detailed_report: str
    final_output: Dict[str, Any]


class MultiStageResearchWorkflow:
    """
    Advanced multi-stage research workflow with configurable stages.
    
    Features:
    - Configurable stage execution (enable/disable stages)
    - Parallel source collection
    - Integrated citation management
    - Comprehensive error handling
    - Progress tracking and metrics
    """
    
    def __init__(
        self,
        llm_client: Any = None,
        academic_search_client: Any = None,
        web_search_client: Any = None,
        google_scholar_client: Any = None,
        citation_manager: Any = None,
        checkpointer: Optional[MemorySaver] = None,
    ):
        self.llm_client = llm_client
        self.academic_search = academic_search_client
        self.web_search = web_search_client
        self.google_scholar = google_scholar_client
        self.citation_manager = citation_manager
        self.checkpointer = checkpointer or MemorySaver()
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the multi-stage workflow graph"""
        workflow = StateGraph(MultiStageState)
        
        # Add stage nodes
        workflow.add_node("discovery", self._discovery_stage)
        workflow.add_node("collection", self._collection_stage)
        workflow.add_node("analysis", self._analysis_stage)
        workflow.add_node("verification", self._verification_stage)
        workflow.add_node("synthesis", self._synthesis_stage)
        workflow.add_node("report", self._report_stage)
        
        # Define flow
        workflow.add_edge(START, "discovery")
        workflow.add_edge("discovery", "collection")
        workflow.add_edge("collection", "analysis")
        
        # Conditional verification
        workflow.add_conditional_edges(
            "analysis",
            self._should_verify,
            {True: "verification", False: "synthesis"}
        )
        
        workflow.add_edge("verification", "synthesis")
        workflow.add_edge("synthesis", "report")
        workflow.add_edge("report", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _should_verify(self, state: MultiStageState) -> bool:
        """Determine if verification stage should run"""
        return state.get("use_fact_verification", False)
    
    async def _discovery_stage(self, state: MultiStageState) -> Dict[str, Any]:
        """
        Stage 1: Discovery
        
        Analyzes the research topic, identifies key concepts,
        defines research scope, and creates search strategy.
        """
        logger.info(f"[{state.get('request_id')}] Starting Discovery stage")
        start_time = datetime.now()
        
        topic = state["topic"]
        research_question = state["research_question"]
        discipline = state.get("discipline", "general")
        
        # Topic analysis
        topic_analysis = await self._analyze_topic(topic, research_question, discipline)
        
        # Extract key concepts
        key_concepts = topic_analysis.get("key_concepts", [])
        if not key_concepts:
            # Fallback extraction
            words = topic.lower().split()
            stop_words = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "how", "what", "why"}
            key_concepts = [w for w in words if w not in stop_words and len(w) > 3][:10]
        
        # Define research scope
        research_scope = {
            "primary_focus": topic,
            "secondary_topics": key_concepts[:5],
            "discipline": discipline,
            "time_scope": "recent",  # Could be configurable
            "geographic_scope": "global",
        }
        
        # Create search strategy
        search_strategy = self._create_search_strategy(
            topic, research_question, key_concepts, discipline
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "current_stage": ResearchStage.DISCOVERY.value,
            "topic_analysis": topic_analysis,
            "key_concepts": key_concepts,
            "research_scope": research_scope,
            "search_strategy": search_strategy,
            "stage_results": {
                **state.get("stage_results", {}),
                ResearchStage.DISCOVERY.value: {
                    "status": StageStatus.COMPLETED.value,
                    "duration_seconds": duration,
                    "metrics": {
                        "concepts_identified": len(key_concepts),
                        "queries_generated": len(search_strategy.get("queries", [])),
                    }
                }
            }
        }
    
    async def _collection_stage(self, state: MultiStageState) -> Dict[str, Any]:
        """
        Stage 2: Collection
        
        Gathers sources from multiple databases in parallel:
        - Academic papers (OpenAlex, CrossRef, Semantic Scholar)
        - Google Scholar
        - Web sources
        """
        logger.info(f"[{state.get('request_id')}] Starting Collection stage")
        start_time = datetime.now()
        
        search_strategy = state.get("search_strategy", {})
        queries = search_strategy.get("queries", [state["topic"]])
        max_sources = state.get("max_sources", 50)
        
        # Parallel collection tasks
        tasks = []
        
        if state.get("use_academic_search", True) and self.academic_search:
            tasks.append(self._collect_academic_papers(queries, max_sources // 3))
        
        if state.get("use_google_scholar", True) and self.google_scholar:
            tasks.append(self._collect_google_scholar(queries, max_sources // 3))
        
        if state.get("use_web_search", False) and self.web_search:
            tasks.append(self._collect_web_sources(queries, max_sources // 3))
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        academic_papers = []
        google_scholar_papers = []
        web_sources = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Collection error: {str(result)}")
                continue
            
            if i == 0 and state.get("use_academic_search", True):
                academic_papers = result
            elif i == 1 and state.get("use_google_scholar", True):
                google_scholar_papers = result
            elif state.get("use_web_search", False):
                web_sources = result
        
        # Combine all sources
        all_sources = []
        for paper in academic_papers:
            all_sources.append({**paper, "source_type": "academic"})
        for paper in google_scholar_papers:
            all_sources.append({**paper, "source_type": "google_scholar"})
        for source in web_sources:
            all_sources.append({**source, "source_type": "web"})
        
        # Deduplicate by title similarity
        all_sources = self._deduplicate_sources(all_sources)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "current_stage": ResearchStage.COLLECTION.value,
            "academic_papers": academic_papers,
            "google_scholar_papers": google_scholar_papers,
            "web_sources": web_sources,
            "all_sources": all_sources,
            "errors": errors,
            "stage_results": {
                **state.get("stage_results", {}),
                ResearchStage.COLLECTION.value: {
                    "status": StageStatus.COMPLETED.value,
                    "duration_seconds": duration,
                    "metrics": {
                        "academic_papers": len(academic_papers),
                        "google_scholar_papers": len(google_scholar_papers),
                        "web_sources": len(web_sources),
                        "total_sources": len(all_sources),
                    }
                }
            }
        }
    
    async def _analysis_stage(self, state: MultiStageState) -> Dict[str, Any]:
        """
        Stage 3: Analysis
        
        Performs deep analysis of collected sources:
        - Source quality assessment
        - Theme extraction
        - Methodology analysis
        - Trend identification
        """
        logger.info(f"[{state.get('request_id')}] Starting Analysis stage")
        start_time = datetime.now()
        
        all_sources = state.get("all_sources", [])
        topic = state["topic"]
        research_question = state["research_question"]
        
        # Source analysis
        source_analysis = await self._analyze_sources(all_sources)
        
        # Theme analysis
        theme_analysis = await self._extract_themes(all_sources, topic, research_question)
        
        # Methodology analysis
        methodology_analysis = self._analyze_methodologies(all_sources, state.get("research_type", "systematic_review"))
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "current_stage": ResearchStage.ANALYSIS.value,
            "source_analysis": source_analysis,
            "theme_analysis": theme_analysis,
            "methodology_analysis": methodology_analysis,
            "stage_results": {
                **state.get("stage_results", {}),
                ResearchStage.ANALYSIS.value: {
                    "status": StageStatus.COMPLETED.value,
                    "duration_seconds": duration,
                    "metrics": {
                        "sources_analyzed": len(all_sources),
                        "themes_identified": len(theme_analysis.get("themes", [])),
                    }
                }
            }
        }
    
    async def _verification_stage(self, state: MultiStageState) -> Dict[str, Any]:
        """
        Stage 4: Verification (Optional)
        
        Verifies key claims and assesses source credibility.
        """
        logger.info(f"[{state.get('request_id')}] Starting Verification stage")
        start_time = datetime.now()
        
        theme_analysis = state.get("theme_analysis", {})
        key_findings = theme_analysis.get("key_findings", [])
        
        verified_claims = []
        for finding in key_findings[:5]:  # Limit to top 5 claims
            verification = await self._verify_claim(finding, state.get("all_sources", []))
            verified_claims.append(verification)
        
        # Assess source credibility
        source_credibility = self._assess_source_credibility(state.get("all_sources", []))
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "current_stage": ResearchStage.VERIFICATION.value,
            "verified_claims": verified_claims,
            "source_credibility": source_credibility,
            "stage_results": {
                **state.get("stage_results", {}),
                ResearchStage.VERIFICATION.value: {
                    "status": StageStatus.COMPLETED.value,
                    "duration_seconds": duration,
                    "metrics": {
                        "claims_verified": len(verified_claims),
                        "average_credibility": source_credibility.get("average_score", 0),
                    }
                }
            }
        }
    
    async def _synthesis_stage(self, state: MultiStageState) -> Dict[str, Any]:
        """
        Stage 5: Synthesis
        
        Integrates all analysis into coherent knowledge:
        - Cross-source synthesis
        - Gap identification
        - Recommendation generation
        """
        logger.info(f"[{state.get('request_id')}] Starting Synthesis stage")
        start_time = datetime.now()
        
        # Synthesize knowledge
        synthesized_knowledge = await self._synthesize_knowledge(
            topic=state["topic"],
            research_question=state["research_question"],
            sources=state.get("all_sources", []),
            theme_analysis=state.get("theme_analysis", {}),
            source_analysis=state.get("source_analysis", {}),
        )
        
        # Identify knowledge gaps
        knowledge_gaps = self._identify_gaps(
            state.get("topic_analysis", {}),
            state.get("theme_analysis", {}),
            state.get("all_sources", [])
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            synthesized_knowledge,
            knowledge_gaps,
            state.get("research_type", "systematic_review")
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "current_stage": ResearchStage.SYNTHESIS.value,
            "synthesized_knowledge": synthesized_knowledge,
            "knowledge_gaps": knowledge_gaps,
            "recommendations": recommendations,
            "stage_results": {
                **state.get("stage_results", {}),
                ResearchStage.SYNTHESIS.value: {
                    "status": StageStatus.COMPLETED.value,
                    "duration_seconds": duration,
                    "metrics": {
                        "gaps_identified": len(knowledge_gaps),
                        "recommendations_generated": len(recommendations),
                    }
                }
            }
        }
    
    async def _report_stage(self, state: MultiStageState) -> Dict[str, Any]:
        """
        Stage 6: Report
        
        Generates comprehensive research report with citations.
        """
        logger.info(f"[{state.get('request_id')}] Starting Report stage")
        start_time = datetime.now()
        
        # Generate citations
        citations = self._generate_citations(
            state.get("all_sources", []),
            state.get("citation_style", "apa7")
        )
        
        # Generate bibliography
        bibliography = self._generate_bibliography(citations, state.get("citation_style", "apa7"))
        
        # Generate executive summary
        executive_summary = await self._generate_executive_summary(
            topic=state["topic"],
            research_question=state["research_question"],
            synthesized_knowledge=state.get("synthesized_knowledge", {}),
            recommendations=state.get("recommendations", [])
        )
        
        # Generate detailed report
        detailed_report = await self._generate_detailed_report(state)
        
        # Compile final output
        final_output = {
            "request_id": state.get("request_id", ""),
            "created_at": state.get("created_at", ""),
            "completed_at": datetime.now().isoformat(),
            "topic": state["topic"],
            "research_question": state["research_question"],
            "research_type": state.get("research_type", "systematic_review"),
            "discipline": state.get("discipline", "general"),
            "executive_summary": executive_summary,
            "detailed_report": detailed_report,
            "key_findings": state.get("synthesized_knowledge", {}).get("key_findings", []),
            "themes": state.get("theme_analysis", {}).get("themes", []),
            "knowledge_gaps": state.get("knowledge_gaps", []),
            "recommendations": state.get("recommendations", []),
            "methodology": state.get("methodology_analysis", {}),
            "sources_count": len(state.get("all_sources", [])),
            "citations": citations,
            "bibliography": bibliography,
            "stage_results": state.get("stage_results", {}),
            "quality_metrics": self._calculate_quality_metrics(state),
        }
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "current_stage": ResearchStage.REPORT.value,
            "citations": citations,
            "bibliography": bibliography,
            "executive_summary": executive_summary,
            "detailed_report": detailed_report,
            "final_output": final_output,
            "stage_results": {
                **state.get("stage_results", {}),
                ResearchStage.REPORT.value: {
                    "status": StageStatus.COMPLETED.value,
                    "duration_seconds": duration,
                    "metrics": {
                        "citations_generated": len(citations),
                        "report_length": len(detailed_report),
                    }
                }
            }
        }
    
    # Helper methods
    
    async def _analyze_topic(self, topic: str, research_question: str, discipline: str) -> Dict[str, Any]:
        """Analyze research topic using LLM or rule-based approach"""
        if self.llm_client:
            try:
                # Use LLM for analysis
                prompt = f"""Analyze this research topic and provide structured insights:

Topic: {topic}
Research Question: {research_question}
Discipline: {discipline}

Provide:
1. Key concepts (list of important terms)
2. Research scope (narrow/moderate/broad)
3. Complexity level (basic/intermediate/advanced)
4. Related disciplines
5. Potential challenges
"""
                # This would call the LLM
                pass
            except Exception as e:
                logger.warning(f"LLM topic analysis failed: {e}")
        
        # Fallback to rule-based
        words = topic.lower().split()
        stop_words = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or"}
        key_concepts = [w for w in words if w not in stop_words and len(w) > 3]
        
        return {
            "topic": topic,
            "research_question": research_question,
            "key_concepts": key_concepts[:10],
            "scope": "moderate",
            "complexity": "intermediate",
            "related_disciplines": [discipline],
            "challenges": ["Data availability", "Scope definition"],
        }
    
    def _create_search_strategy(
        self,
        topic: str,
        research_question: str,
        key_concepts: List[str],
        discipline: str
    ) -> Dict[str, Any]:
        """Create optimized search strategy"""
        queries = []
        
        # Primary query
        queries.append(topic)
        
        # Research question as query
        queries.append(research_question)
        
        # Concept combinations
        for concept in key_concepts[:5]:
            queries.append(f"{topic} {concept}")
        
        # Discipline-specific query
        queries.append(f"{topic} {discipline}")
        
        return {
            "queries": queries,
            "filters": {
                "discipline": discipline,
                "recency": "5_years",
            },
            "priority_sources": ["academic", "google_scholar"],
        }
    
    async def _collect_academic_papers(self, queries: List[str], limit: int) -> List[Dict[str, Any]]:
        """Collect papers from academic databases"""
        papers = []
        
        if self.academic_search:
            for query in queries[:3]:
                try:
                    results = await self.academic_search.execute(query=query, num_results=limit // 3)
                    for r in results:
                        papers.append(r.model_dump() if hasattr(r, 'model_dump') else r)
                except Exception as e:
                    logger.warning(f"Academic search failed for '{query}': {e}")
        
        return papers
    
    async def _collect_google_scholar(self, queries: List[str], limit: int) -> List[Dict[str, Any]]:
        """Collect papers from Google Scholar"""
        papers = []
        
        if self.google_scholar:
            for query in queries[:2]:
                try:
                    results = await self.google_scholar.search(query=query, num_results=limit // 2)
                    for r in results:
                        papers.append(r.model_dump() if hasattr(r, 'model_dump') else r)
                except Exception as e:
                    logger.warning(f"Google Scholar search failed for '{query}': {e}")
        
        return papers
    
    async def _collect_web_sources(self, queries: List[str], limit: int) -> List[Dict[str, Any]]:
        """Collect sources from web search"""
        sources = []
        
        if self.web_search:
            for query in queries[:2]:
                try:
                    results = await self.web_search.execute(query=query, num_results=limit // 2)
                    for r in results:
                        sources.append(r.model_dump() if hasattr(r, 'model_dump') else r)
                except Exception as e:
                    logger.warning(f"Web search failed for '{query}': {e}")
        
        return sources
    
    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate sources based on title similarity"""
        seen_titles = set()
        unique = []
        
        for source in sources:
            title = source.get("title", "").lower().strip()
            # Simple deduplication by exact title match
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique.append(source)
            elif not title:
                unique.append(source)
        
        return unique
    
    async def _analyze_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze collected sources"""
        if not sources:
            return {"total": 0, "by_type": {}, "by_year": {}}
        
        by_type = {}
        by_year = {}
        
        for source in sources:
            # Count by type
            source_type = source.get("source_type", "unknown")
            by_type[source_type] = by_type.get(source_type, 0) + 1
            
            # Count by year
            year = source.get("year")
            if year:
                by_year[str(year)] = by_year.get(str(year), 0) + 1
        
        return {
            "total": len(sources),
            "by_type": by_type,
            "by_year": by_year,
            "year_range": f"{min(by_year.keys()) if by_year else 'N/A'}-{max(by_year.keys()) if by_year else 'N/A'}",
        }
    
    async def _extract_themes(
        self,
        sources: List[Dict[str, Any]],
        topic: str,
        research_question: str
    ) -> Dict[str, Any]:
        """Extract themes from sources"""
        # This would use LLM for better theme extraction
        # For now, use simple keyword extraction
        
        all_text = " ".join([
            source.get("abstract", "") or source.get("snippet", "") or ""
            for source in sources
        ]).lower()
        
        # Simple theme extraction based on frequency
        words = all_text.split()
        word_freq = {}
        stop_words = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "this", "that", "with", "from", "by", "as", "be", "was", "were", "been", "have", "has", "had"}
        
        for word in words:
            word = word.strip(".,;:!?()[]{}\"'")
            if len(word) > 4 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top themes
        themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "themes": [t[0] for t in themes],
            "theme_frequencies": dict(themes),
            "key_findings": [f"Research focuses on {themes[0][0]}" if themes else "Insufficient data for theme extraction"],
        }
    
    def _analyze_methodologies(self, sources: List[Dict[str, Any]], research_type: str) -> Dict[str, Any]:
        """Analyze methodologies used in sources"""
        methodology_keywords = {
            "quantitative": ["survey", "experiment", "statistical", "regression", "correlation", "sample size"],
            "qualitative": ["interview", "case study", "ethnography", "grounded theory", "thematic analysis"],
            "mixed_methods": ["mixed method", "triangulation", "sequential", "concurrent"],
            "systematic_review": ["systematic review", "meta-analysis", "prisma", "inclusion criteria"],
        }
        
        methodology_counts = {k: 0 for k in methodology_keywords}
        
        for source in sources:
            text = (source.get("abstract", "") or source.get("snippet", "") or "").lower()
            for method, keywords in methodology_keywords.items():
                if any(kw in text for kw in keywords):
                    methodology_counts[method] += 1
        
        dominant = max(methodology_counts.items(), key=lambda x: x[1])[0] if any(methodology_counts.values()) else "unknown"
        
        return {
            "methodology_distribution": methodology_counts,
            "dominant_methodology": dominant,
            "recommended_approach": research_type,
        }
    
    async def _verify_claim(self, claim: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify a claim against sources"""
        # Simplified verification - would use LLM for better results
        supporting = 0
        contradicting = 0
        
        claim_words = set(claim.lower().split())
        
        for source in sources:
            text = (source.get("abstract", "") or source.get("snippet", "") or "").lower()
            text_words = set(text.split())
            
            overlap = len(claim_words & text_words) / len(claim_words) if claim_words else 0
            if overlap > 0.3:
                supporting += 1
        
        return {
            "claim": claim,
            "supporting_sources": supporting,
            "contradicting_sources": contradicting,
            "confidence": min(supporting / max(len(sources), 1), 1.0),
            "verdict": "supported" if supporting > 2 else "insufficient_evidence",
        }
    
    def _assess_source_credibility(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess credibility of sources"""
        scores = []
        
        for source in sources:
            score = 0.5  # Base score
            
            # Academic sources get higher base score
            if source.get("source_type") == "academic":
                score += 0.2
            elif source.get("source_type") == "google_scholar":
                score += 0.15
            
            # Citation count bonus
            citations = source.get("citation_count", 0)
            if citations and citations > 100:
                score += 0.2
            elif citations and citations > 10:
                score += 0.1
            
            # Recency bonus
            year = source.get("year")
            if year and year >= 2020:
                score += 0.1
            
            scores.append(min(score, 1.0))
        
        return {
            "average_score": sum(scores) / len(scores) if scores else 0,
            "high_credibility_count": sum(1 for s in scores if s > 0.7),
            "low_credibility_count": sum(1 for s in scores if s < 0.4),
        }
    
    async def _synthesize_knowledge(
        self,
        topic: str,
        research_question: str,
        sources: List[Dict[str, Any]],
        theme_analysis: Dict[str, Any],
        source_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize knowledge from all analyses"""
        themes = theme_analysis.get("themes", [])
        
        return {
            "topic": topic,
            "research_question": research_question,
            "key_findings": [
                f"Analysis of {source_analysis.get('total', 0)} sources reveals focus on {', '.join(themes[:3]) if themes else 'the research topic'}",
                f"Research spans {source_analysis.get('year_range', 'multiple years')}",
            ],
            "themes": themes,
            "consensus_points": [f"Common focus on {themes[0]}" if themes else "Limited consensus identified"],
            "controversies": [],
            "confidence_level": "medium" if len(sources) > 10 else "low",
        }
    
    def _identify_gaps(
        self,
        topic_analysis: Dict[str, Any],
        theme_analysis: Dict[str, Any],
        sources: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify knowledge gaps"""
        gaps = []
        
        # Check for underrepresented concepts
        key_concepts = topic_analysis.get("key_concepts", [])
        themes = theme_analysis.get("themes", [])
        
        for concept in key_concepts:
            if concept not in themes:
                gaps.append(f"Limited research on '{concept}' aspect")
        
        # Check for recency
        years = [s.get("year") for s in sources if s.get("year")]
        if years and max(years) < 2022:
            gaps.append("Lack of recent research (post-2022)")
        
        # Check for methodology gaps
        if len(sources) < 10:
            gaps.append("Limited number of sources available")
        
        return gaps[:5]  # Limit to top 5 gaps
    
    def _generate_recommendations(
        self,
        synthesized_knowledge: Dict[str, Any],
        knowledge_gaps: List[str],
        research_type: str
    ) -> List[str]:
        """Generate research recommendations"""
        recommendations = []
        
        # Based on gaps
        for gap in knowledge_gaps[:3]:
            recommendations.append(f"Address gap: {gap}")
        
        # Based on research type
        if research_type == "systematic_review":
            recommendations.append("Follow PRISMA guidelines for systematic review")
        elif research_type == "meta_analysis":
            recommendations.append("Ensure sufficient quantitative studies for meta-analysis")
        
        # General recommendations
        recommendations.append("Consider multi-disciplinary perspectives")
        recommendations.append("Validate findings with domain experts")
        
        return recommendations
    
    def _generate_citations(self, sources: List[Dict[str, Any]], style: str) -> List[Dict[str, Any]]:
        """Generate citations for all sources"""
        citations = []
        
        for i, source in enumerate(sources):
            citation = {
                "id": i + 1,
                "title": source.get("title", "Untitled"),
                "authors": source.get("authors", ["Unknown"]),
                "year": source.get("year"),
                "source_type": source.get("source_type", "unknown"),
                "url": source.get("url"),
                "doi": source.get("doi"),
            }
            citations.append(citation)
        
        return citations
    
    def _generate_bibliography(self, citations: List[Dict[str, Any]], style: str) -> str:
        """Generate formatted bibliography"""
        if self.citation_manager:
            # Use citation manager for proper formatting
            for citation in citations:
                self.citation_manager.add_from_dict(citation)
            return self.citation_manager.generate_bibliography()
        
        # Simple fallback
        lines = []
        for c in citations:
            authors = ", ".join(c.get("authors", ["Unknown"])[:3])
            year = c.get("year", "n.d.")
            title = c.get("title", "Untitled")
            lines.append(f"{authors} ({year}). {title}.")
        
        return "\n\n".join(lines)
    
    async def _generate_executive_summary(
        self,
        topic: str,
        research_question: str,
        synthesized_knowledge: Dict[str, Any],
        recommendations: List[str]
    ) -> str:
        """Generate executive summary"""
        findings = synthesized_knowledge.get("key_findings", [])
        themes = synthesized_knowledge.get("themes", [])
        
        summary = f"""# Executive Summary

## Research Topic
{topic}

## Research Question
{research_question}

## Key Findings
{chr(10).join(f"- {f}" for f in findings)}

## Major Themes
{chr(10).join(f"- {t}" for t in themes[:5])}

## Recommendations
{chr(10).join(f"- {r}" for r in recommendations[:5])}
"""
        return summary
    
    async def _generate_detailed_report(self, state: MultiStageState) -> str:
        """Generate detailed research report"""
        report = f"""# Research Report: {state['topic']}

## 1. Introduction

### Research Question
{state['research_question']}

### Scope
{state.get('research_scope', {}).get('primary_focus', state['topic'])}

## 2. Methodology

### Search Strategy
- Queries used: {len(state.get('search_strategy', {}).get('queries', []))}
- Sources searched: Academic databases, Google Scholar

### Source Analysis
- Total sources: {state.get('source_analysis', {}).get('total', 0)}
- Year range: {state.get('source_analysis', {}).get('year_range', 'N/A')}

## 3. Findings

### Key Themes
{chr(10).join(f"- {t}" for t in state.get('theme_analysis', {}).get('themes', [])[:10])}

### Key Findings
{chr(10).join(f"- {f}" for f in state.get('synthesized_knowledge', {}).get('key_findings', []))}

## 4. Knowledge Gaps
{chr(10).join(f"- {g}" for g in state.get('knowledge_gaps', []))}

## 5. Recommendations
{chr(10).join(f"- {r}" for r in state.get('recommendations', []))}

## 6. Conclusion

This research analysis examined {state.get('source_analysis', {}).get('total', 0)} sources 
related to "{state['topic']}". The analysis identified {len(state.get('theme_analysis', {}).get('themes', []))} 
major themes and {len(state.get('knowledge_gaps', []))} knowledge gaps requiring further investigation.

## References

{state.get('bibliography', 'See attached bibliography.')}
"""
        return report
    
    def _calculate_quality_metrics(self, state: MultiStageState) -> Dict[str, Any]:
        """Calculate overall quality metrics"""
        sources = state.get("all_sources", [])
        stages = state.get("stage_results", {})
        
        # Calculate completion rate
        completed_stages = sum(1 for s in stages.values() if s.get("status") == StageStatus.COMPLETED.value)
        total_stages = 6
        
        # Calculate source quality
        academic_ratio = len(state.get("academic_papers", [])) / max(len(sources), 1)
        
        return {
            "completion_rate": completed_stages / total_stages,
            "source_count": len(sources),
            "academic_source_ratio": academic_ratio,
            "themes_identified": len(state.get("theme_analysis", {}).get("themes", [])),
            "gaps_identified": len(state.get("knowledge_gaps", [])),
            "overall_quality": "high" if len(sources) > 20 and academic_ratio > 0.5 else "medium" if len(sources) > 10 else "low",
        }
    
    async def run(
        self,
        topic: str,
        research_question: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the multi-stage research workflow.
        
        Args:
            topic: Research topic
            research_question: Main research question
            **kwargs: Additional configuration
        
        Returns:
            Complete research output
        """
        request_id = kwargs.get("request_id", str(uuid.uuid4()))
        
        initial_state: MultiStageState = {
            "request_id": request_id,
            "created_at": datetime.now().isoformat(),
            "topic": topic,
            "research_question": research_question,
            "research_type": kwargs.get("research_type", "systematic_review"),
            "discipline": kwargs.get("discipline", "general"),
            "academic_level": kwargs.get("academic_level", "graduate"),
            "additional_context": kwargs.get("additional_context"),
            "use_llm": kwargs.get("use_llm", True),
            "use_web_search": kwargs.get("use_web_search", False),
            "use_academic_search": kwargs.get("use_academic_search", True),
            "use_google_scholar": kwargs.get("use_google_scholar", True),
            "use_fact_verification": kwargs.get("use_fact_verification", False),
            "citation_style": kwargs.get("citation_style", "apa7"),
            "max_sources": kwargs.get("max_sources", 50),
            "stages_config": kwargs.get("stages_config", {}),
            "stage_results": {},
            "errors": [],
        }
        
        config = {"configurable": {"thread_id": request_id}}
        
        logger.info(f"Starting multi-stage research workflow: {request_id}")
        
        result = await self.graph.ainvoke(initial_state, config)
        
        logger.info(f"Completed multi-stage research workflow: {request_id}")
        
        return result.get("final_output", result)


def create_multi_stage_workflow(
    llm_client: Any = None,
    academic_search_client: Any = None,
    web_search_client: Any = None,
    google_scholar_client: Any = None,
    citation_manager: Any = None,
) -> MultiStageResearchWorkflow:
    """Factory function to create multi-stage workflow"""
    return MultiStageResearchWorkflow(
        llm_client=llm_client,
        academic_search_client=academic_search_client,
        web_search_client=web_search_client,
        google_scholar_client=google_scholar_client,
        citation_manager=citation_manager,
    )
