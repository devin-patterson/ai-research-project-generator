"""
Research Tools for AI-Powered Research Workflows

This module provides a comprehensive set of tools for gathering, synthesizing,
and verifying research information. Tools are designed to work with both
LangGraph and PydanticAI agent frameworks.

Tools follow these patterns:
- LangGraph: @tool decorator from langchain_core.tools
- PydanticAI: @agent.tool decorator or FunctionToolset

Author: AI Research Generator
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

import httpx
from loguru import logger
from pydantic import BaseModel, Field

# Try to import langchain tools for LangGraph compatibility
try:
    from langchain_core.tools import tool as langchain_tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    def langchain_tool(func):
        """Fallback decorator when langchain is not available"""
        return func


# =============================================================================
# Tool Output Schemas
# =============================================================================


class WebSearchResult(BaseModel):
    """Schema for web search results"""
    title: str = Field(description="Title of the web page")
    url: str = Field(description="URL of the web page")
    snippet: str = Field(description="Text snippet from the page")
    source: str = Field(description="Source domain")
    relevance_score: float = Field(default=0.0, description="Relevance score 0-1")


class AcademicPaperResult(BaseModel):
    """Schema for academic paper search results"""
    title: str = Field(description="Paper title")
    authors: List[str] = Field(description="List of authors")
    abstract: Optional[str] = Field(default=None, description="Paper abstract")
    year: Optional[int] = Field(default=None, description="Publication year")
    doi: Optional[str] = Field(default=None, description="DOI identifier")
    url: Optional[str] = Field(default=None, description="URL to paper")
    citation_count: Optional[int] = Field(default=None, description="Number of citations")
    source: str = Field(description="Source database (e.g., OpenAlex, CrossRef)")


class SynthesizedKnowledge(BaseModel):
    """Schema for synthesized knowledge output"""
    topic: str = Field(description="Research topic")
    key_findings: List[str] = Field(description="Key findings from research")
    themes: List[str] = Field(description="Major themes identified")
    consensus_points: List[str] = Field(description="Points of agreement across sources")
    controversies: List[str] = Field(default_factory=list, description="Areas of disagreement")
    gaps: List[str] = Field(default_factory=list, description="Identified knowledge gaps")
    recommendations: List[str] = Field(description="Actionable recommendations")
    confidence_level: str = Field(description="Overall confidence in findings (low/medium/high)")
    sources_used: int = Field(description="Number of sources synthesized")


class FactVerificationResult(BaseModel):
    """Schema for fact verification output"""
    claim: str = Field(description="The claim being verified")
    verdict: str = Field(description="Verification verdict (verified/unverified/disputed/insufficient_evidence)")
    confidence: float = Field(description="Confidence score 0-1")
    supporting_evidence: List[str] = Field(default_factory=list, description="Evidence supporting the claim")
    contradicting_evidence: List[str] = Field(default_factory=list, description="Evidence contradicting the claim")
    sources: List[str] = Field(description="Sources consulted for verification")
    explanation: str = Field(description="Detailed explanation of the verdict")


class ResearchReport(BaseModel):
    """Schema for comprehensive research report"""
    title: str = Field(description="Report title")
    executive_summary: str = Field(description="Executive summary of findings")
    research_question: str = Field(description="The research question addressed")
    methodology: str = Field(description="Research methodology used")
    key_findings: List[str] = Field(description="Key findings")
    detailed_analysis: str = Field(description="Detailed analysis and discussion")
    recommendations: List[str] = Field(description="Actionable recommendations")
    limitations: List[str] = Field(description="Research limitations")
    future_research: List[str] = Field(description="Suggestions for future research")
    references: List[str] = Field(description="References and sources")
    generated_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Tool Configuration
# =============================================================================


@dataclass
class ToolConfig:
    """Configuration for research tools"""
    # HTTP settings
    http_timeout: float = 30.0
    max_retries: int = 3
    
    # Search settings
    max_web_results: int = 10
    max_academic_results: int = 20
    
    # API endpoints
    serper_api_key: Optional[str] = field(default_factory=lambda: os.getenv("SERPER_API_KEY"))
    tavily_api_key: Optional[str] = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    
    # LLM settings for synthesis
    llm_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))


# =============================================================================
# Base Tool Class
# =============================================================================


class BaseTool(ABC):
    """Abstract base class for research tools"""
    
    def __init__(self, config: Optional[ToolConfig] = None):
        self.config = config or ToolConfig()
        self._http_client: Optional[httpx.AsyncClient] = None
    
    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.config.http_timeout)
        return self._http_client
    
    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for registration"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM"""
        pass


# =============================================================================
# Web Search Tool
# =============================================================================


class WebSearchTool(BaseTool):
    """Tool for searching the web for information"""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for information on a given query. Returns relevant web pages with titles, URLs, and snippets."
    
    async def execute(
        self,
        query: str,
        num_results: int = 10,
        search_type: str = "general"
    ) -> List[WebSearchResult]:
        """
        Execute web search.
        
        Args:
            query: Search query string
            num_results: Maximum number of results to return
            search_type: Type of search (general, news, academic)
        
        Returns:
            List of WebSearchResult objects
        """
        logger.info(f"Executing web search: {query[:50]}...")
        
        results = []
        
        # Try Serper API first (Google Search)
        if self.config.serper_api_key:
            try:
                results = await self._search_serper(query, num_results)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Serper search failed: {e}")
        
        # Try Tavily API as fallback
        if self.config.tavily_api_key:
            try:
                results = await self._search_tavily(query, num_results)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}")
        
        # Fallback to DuckDuckGo (no API key required)
        try:
            results = await self._search_duckduckgo(query, num_results)
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
        
        return results
    
    async def _search_serper(self, query: str, num_results: int) -> List[WebSearchResult]:
        """Search using Serper API (Google Search)"""
        response = await self.http_client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self.config.serper_api_key},
            json={"q": query, "num": num_results}
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("organic", [])[:num_results]:
            results.append(WebSearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source=item.get("link", "").split("/")[2] if item.get("link") else "",
                relevance_score=1.0 - (len(results) * 0.05)  # Decay by position
            ))
        return results
    
    async def _search_tavily(self, query: str, num_results: int) -> List[WebSearchResult]:
        """Search using Tavily API"""
        response = await self.http_client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.config.tavily_api_key,
                "query": query,
                "max_results": num_results,
                "search_depth": "advanced"
            }
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("results", [])[:num_results]:
            results.append(WebSearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source=item.get("url", "").split("/")[2] if item.get("url") else "",
                relevance_score=item.get("score", 0.5)
            ))
        return results
    
    async def _search_duckduckgo(self, query: str, num_results: int) -> List[WebSearchResult]:
        """Search using DuckDuckGo (no API key required)"""
        # Use DuckDuckGo instant answer API
        response = await self.http_client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1}
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        
        # Get related topics
        for item in data.get("RelatedTopics", [])[:num_results]:
            if isinstance(item, dict) and "FirstURL" in item:
                results.append(WebSearchResult(
                    title=item.get("Text", "")[:100],
                    url=item.get("FirstURL", ""),
                    snippet=item.get("Text", ""),
                    source="duckduckgo.com",
                    relevance_score=0.7
                ))
        
        return results


# =============================================================================
# Academic Search Tool
# =============================================================================


class AcademicSearchTool(BaseTool):
    """Tool for searching academic papers and publications"""
    
    @property
    def name(self) -> str:
        return "academic_search"
    
    @property
    def description(self) -> str:
        return "Search academic databases for scholarly papers and publications. Returns papers with titles, authors, abstracts, and citations."
    
    async def execute(
        self,
        query: str,
        num_results: int = 20,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None
    ) -> List[AcademicPaperResult]:
        """
        Execute academic paper search.
        
        Args:
            query: Search query string
            num_results: Maximum number of results
            year_from: Filter papers from this year
            year_to: Filter papers up to this year
        
        Returns:
            List of AcademicPaperResult objects
        """
        logger.info(f"Executing academic search: {query[:50]}...")
        
        all_results = []
        
        # Search OpenAlex
        try:
            openalex_results = await self._search_openalex(query, num_results, year_from, year_to)
            all_results.extend(openalex_results)
        except Exception as e:
            logger.warning(f"OpenAlex search failed: {e}")
        
        # Search CrossRef
        try:
            crossref_results = await self._search_crossref(query, num_results, year_from, year_to)
            all_results.extend(crossref_results)
        except Exception as e:
            logger.warning(f"CrossRef search failed: {e}")
        
        # Search Semantic Scholar
        try:
            semantic_results = await self._search_semantic_scholar(query, num_results)
            all_results.extend(semantic_results)
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")
        
        # Deduplicate by DOI
        seen_dois = set()
        unique_results = []
        for paper in all_results:
            if paper.doi and paper.doi in seen_dois:
                continue
            if paper.doi:
                seen_dois.add(paper.doi)
            unique_results.append(paper)
        
        # Sort by citation count
        unique_results.sort(key=lambda x: x.citation_count or 0, reverse=True)
        
        return unique_results[:num_results]
    
    async def _search_openalex(
        self,
        query: str,
        num_results: int,
        year_from: Optional[int],
        year_to: Optional[int]
    ) -> List[AcademicPaperResult]:
        """Search OpenAlex database"""
        params = {
            "search": query,
            "per_page": min(num_results, 50),
            "sort": "cited_by_count:desc"
        }
        
        if year_from or year_to:
            year_filter = []
            if year_from:
                year_filter.append(f"from_publication_date:{year_from}-01-01")
            if year_to:
                year_filter.append(f"to_publication_date:{year_to}-12-31")
            params["filter"] = ",".join(year_filter)
        
        response = await self.http_client.get(
            "https://api.openalex.org/works",
            params=params,
            headers={"User-Agent": "AIResearchGenerator/1.0"}
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for work in data.get("results", []):
            authors = [
                a.get("author", {}).get("display_name", "Unknown")
                for a in work.get("authorships", [])[:5]
            ]
            
            results.append(AcademicPaperResult(
                title=work.get("title", ""),
                authors=authors,
                abstract=work.get("abstract", None),
                year=work.get("publication_year"),
                doi=work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else None,
                url=work.get("doi") or work.get("id"),
                citation_count=work.get("cited_by_count"),
                source="openalex"
            ))
        
        return results
    
    async def _search_crossref(
        self,
        query: str,
        num_results: int,
        year_from: Optional[int],
        year_to: Optional[int]
    ) -> List[AcademicPaperResult]:
        """Search CrossRef database"""
        params = {
            "query": query,
            "rows": min(num_results, 50),
            "sort": "relevance"
        }
        
        if year_from:
            params["filter"] = f"from-pub-date:{year_from}"
        if year_to:
            params["filter"] = params.get("filter", "") + f",until-pub-date:{year_to}"
        
        response = await self.http_client.get(
            "https://api.crossref.org/works",
            params=params,
            headers={"User-Agent": "AIResearchGenerator/1.0"}
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("message", {}).get("items", []):
            authors = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in item.get("author", [])[:5]
            ]
            
            # Get publication year
            pub_date = item.get("published-print", {}) or item.get("published-online", {})
            year = pub_date.get("date-parts", [[None]])[0][0] if pub_date else None
            
            results.append(AcademicPaperResult(
                title=item.get("title", [""])[0] if item.get("title") else "",
                authors=authors,
                abstract=item.get("abstract"),
                year=year,
                doi=item.get("DOI"),
                url=f"https://doi.org/{item.get('DOI')}" if item.get("DOI") else None,
                citation_count=item.get("is-referenced-by-count"),
                source="crossref"
            ))
        
        return results
    
    async def _search_semantic_scholar(
        self,
        query: str,
        num_results: int
    ) -> List[AcademicPaperResult]:
        """Search Semantic Scholar database"""
        response = await self.http_client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": min(num_results, 100),
                "fields": "title,authors,abstract,year,citationCount,externalIds,url"
            }
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for paper in data.get("data", []):
            authors = [a.get("name", "") for a in paper.get("authors", [])[:5]]
            external_ids = paper.get("externalIds", {})
            
            results.append(AcademicPaperResult(
                title=paper.get("title", ""),
                authors=authors,
                abstract=paper.get("abstract"),
                year=paper.get("year"),
                doi=external_ids.get("DOI"),
                url=paper.get("url"),
                citation_count=paper.get("citationCount"),
                source="semantic_scholar"
            ))
        
        return results


# =============================================================================
# Knowledge Synthesis Tool
# =============================================================================


class KnowledgeSynthesisTool(BaseTool):
    """Tool for synthesizing knowledge from multiple sources"""
    
    @property
    def name(self) -> str:
        return "synthesize_knowledge"
    
    @property
    def description(self) -> str:
        return "Synthesize information from multiple sources into coherent knowledge. Identifies themes, consensus, controversies, and gaps."
    
    async def execute(
        self,
        topic: str,
        sources: List[Dict[str, Any]],
        research_question: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> SynthesizedKnowledge:
        """
        Synthesize knowledge from multiple sources.
        
        Args:
            topic: Research topic
            sources: List of source dictionaries with content
            research_question: Optional specific research question
            additional_context: Optional additional context
        
        Returns:
            SynthesizedKnowledge object with synthesized findings
        """
        logger.info(f"Synthesizing knowledge for: {topic[:50]}...")
        
        # Prepare source content for LLM
        source_texts = []
        for i, source in enumerate(sources[:20], 1):  # Limit to 20 sources
            source_text = f"Source {i}:\n"
            if source.get("title"):
                source_text += f"Title: {source['title']}\n"
            if source.get("content") or source.get("snippet") or source.get("abstract"):
                content = source.get("content") or source.get("snippet") or source.get("abstract")
                source_text += f"Content: {content[:1000]}\n"
            source_texts.append(source_text)
        
        combined_sources = "\n\n".join(source_texts)
        
        # Generate synthesis using LLM
        prompt = f"""Synthesize the following sources to answer the research question.

Topic: {topic}
Research Question: {research_question or f"What are the key insights about {topic}?"}

Additional Context:
{additional_context or "None provided"}

Sources:
{combined_sources}

Provide a comprehensive synthesis that includes:
1. Key Findings: The most important discoveries and insights
2. Major Themes: Recurring themes across sources
3. Consensus Points: Areas where sources agree
4. Controversies: Areas of disagreement or debate
5. Knowledge Gaps: What is missing or needs more research
6. Recommendations: Actionable recommendations based on findings

Format your response as JSON with these exact keys:
{{
    "key_findings": ["finding1", "finding2", ...],
    "themes": ["theme1", "theme2", ...],
    "consensus_points": ["point1", "point2", ...],
    "controversies": ["controversy1", ...],
    "gaps": ["gap1", "gap2", ...],
    "recommendations": ["rec1", "rec2", ...],
    "confidence_level": "low|medium|high"
}}"""

        try:
            synthesis_data = await self._call_llm(prompt)
            
            return SynthesizedKnowledge(
                topic=topic,
                key_findings=synthesis_data.get("key_findings", []),
                themes=synthesis_data.get("themes", []),
                consensus_points=synthesis_data.get("consensus_points", []),
                controversies=synthesis_data.get("controversies", []),
                gaps=synthesis_data.get("gaps", []),
                recommendations=synthesis_data.get("recommendations", []),
                confidence_level=synthesis_data.get("confidence_level", "medium"),
                sources_used=len(sources)
            )
        except Exception as e:
            logger.error(f"Knowledge synthesis failed: {e}")
            # Return minimal synthesis on error
            return SynthesizedKnowledge(
                topic=topic,
                key_findings=["Synthesis failed - please review sources manually"],
                themes=[],
                consensus_points=[],
                controversies=[],
                gaps=["Unable to identify gaps due to synthesis failure"],
                recommendations=["Manual review of sources recommended"],
                confidence_level="low",
                sources_used=len(sources)
            )
    
    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM for synthesis"""
        response = await self.http_client.post(
            f"{self.config.llm_base_url}/api/generate",
            json={
                "model": self.config.llm_model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        
        # Parse JSON response
        response_text = data.get("response", "{}")
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}


# =============================================================================
# Fact Verification Tool
# =============================================================================


class FactVerificationTool(BaseTool):
    """Tool for verifying facts and claims"""
    
    @property
    def name(self) -> str:
        return "verify_facts"
    
    @property
    def description(self) -> str:
        return "Verify the accuracy of a claim or statement by cross-referencing multiple sources."
    
    async def execute(
        self,
        claim: str,
        context: Optional[str] = None
    ) -> FactVerificationResult:
        """
        Verify a claim or statement.
        
        Args:
            claim: The claim to verify
            context: Optional context for the claim
        
        Returns:
            FactVerificationResult with verification details
        """
        logger.info(f"Verifying claim: {claim[:50]}...")
        
        # Search for evidence
        web_tool = WebSearchTool(self.config)
        academic_tool = AcademicSearchTool(self.config)
        
        try:
            # Search web for evidence
            web_results = await web_tool.execute(
                query=f"fact check: {claim}",
                num_results=5
            )
            
            # Search academic sources
            academic_results = await academic_tool.execute(
                query=claim,
                num_results=5
            )
            
            # Combine sources
            all_sources = []
            for r in web_results:
                all_sources.append({"type": "web", "title": r.title, "content": r.snippet, "url": r.url})
            for r in academic_results:
                all_sources.append({"type": "academic", "title": r.title, "content": r.abstract or "", "url": r.url})
            
            # Use LLM to analyze evidence
            verification = await self._analyze_evidence(claim, context, all_sources)
            
            return verification
            
        except Exception as e:
            logger.error(f"Fact verification failed: {e}")
            return FactVerificationResult(
                claim=claim,
                verdict="insufficient_evidence",
                confidence=0.0,
                supporting_evidence=[],
                contradicting_evidence=[],
                sources=[],
                explanation=f"Verification failed due to error: {str(e)}"
            )
        finally:
            await web_tool.close()
            await academic_tool.close()
    
    async def _analyze_evidence(
        self,
        claim: str,
        context: Optional[str],
        sources: List[Dict[str, Any]]
    ) -> FactVerificationResult:
        """Analyze evidence using LLM"""
        source_texts = "\n\n".join([
            f"Source ({s['type']}): {s['title']}\nContent: {s['content'][:500]}"
            for s in sources[:10]
        ])
        
        prompt = f"""Verify the following claim based on the provided sources.

Claim: {claim}
Context: {context or "None provided"}

Sources:
{source_texts}

Analyze the evidence and provide:
1. Verdict: verified, unverified, disputed, or insufficient_evidence
2. Confidence: 0.0 to 1.0
3. Supporting evidence from sources
4. Contradicting evidence from sources
5. Explanation of your verdict

Format as JSON:
{{
    "verdict": "verified|unverified|disputed|insufficient_evidence",
    "confidence": 0.0-1.0,
    "supporting_evidence": ["evidence1", ...],
    "contradicting_evidence": ["evidence1", ...],
    "explanation": "detailed explanation"
}}"""

        response = await self.http_client.post(
            f"{self.config.llm_base_url}/api/generate",
            json={
                "model": self.config.llm_model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        
        try:
            result = json.loads(data.get("response", "{}"))
        except json.JSONDecodeError:
            result = {}
        
        return FactVerificationResult(
            claim=claim,
            verdict=result.get("verdict", "insufficient_evidence"),
            confidence=float(result.get("confidence", 0.5)),
            supporting_evidence=result.get("supporting_evidence", []),
            contradicting_evidence=result.get("contradicting_evidence", []),
            sources=[s.get("url", "") for s in sources if s.get("url")],
            explanation=result.get("explanation", "Unable to analyze evidence")
        )


# =============================================================================
# Research Toolkit (Combines All Tools)
# =============================================================================


class ResearchToolkit:
    """
    Comprehensive toolkit that combines all research tools.
    
    This class provides a unified interface for conducting research
    using multiple tools and generating comprehensive reports.
    """
    
    def __init__(self, config: Optional[ToolConfig] = None):
        self.config = config or ToolConfig()
        self.web_search = WebSearchTool(self.config)
        self.academic_search = AcademicSearchTool(self.config)
        self.knowledge_synthesis = KnowledgeSynthesisTool(self.config)
        self.fact_verification = FactVerificationTool(self.config)
    
    async def close(self):
        """Close all tool connections"""
        await self.web_search.close()
        await self.academic_search.close()
        await self.knowledge_synthesis.close()
        await self.fact_verification.close()
    
    async def conduct_research(
        self,
        topic: str,
        research_question: str,
        additional_context: Optional[str] = None,
        include_web_search: bool = True,
        include_academic_search: bool = True,
        verify_key_claims: bool = False
    ) -> ResearchReport:
        """
        Conduct comprehensive research on a topic.
        
        Args:
            topic: Research topic
            research_question: Specific research question
            additional_context: Additional context or requirements
            include_web_search: Whether to include web search
            include_academic_search: Whether to include academic search
            verify_key_claims: Whether to verify key claims found
        
        Returns:
            ResearchReport with comprehensive findings
        """
        logger.info(f"Conducting comprehensive research on: {topic}")
        
        all_sources = []
        
        # Step 1: Gather sources
        if include_web_search:
            logger.info("Step 1a: Searching web sources...")
            web_results = await self.web_search.execute(
                query=f"{topic} {research_question}",
                num_results=self.config.max_web_results
            )
            for r in web_results:
                all_sources.append({
                    "type": "web",
                    "title": r.title,
                    "content": r.snippet,
                    "url": r.url,
                    "source": r.source
                })
        
        if include_academic_search:
            logger.info("Step 1b: Searching academic sources...")
            academic_results = await self.academic_search.execute(
                query=topic,
                num_results=self.config.max_academic_results
            )
            for r in academic_results:
                all_sources.append({
                    "type": "academic",
                    "title": r.title,
                    "content": r.abstract or "",
                    "url": r.url,
                    "authors": r.authors,
                    "year": r.year,
                    "citations": r.citation_count
                })
        
        logger.info(f"Gathered {len(all_sources)} sources")
        
        # Step 2: Synthesize knowledge
        logger.info("Step 2: Synthesizing knowledge...")
        synthesis = await self.knowledge_synthesis.execute(
            topic=topic,
            sources=all_sources,
            research_question=research_question,
            additional_context=additional_context
        )
        
        # Step 3: Generate comprehensive report
        logger.info("Step 3: Generating research report...")
        report = await self._generate_report(
            topic=topic,
            research_question=research_question,
            synthesis=synthesis,
            sources=all_sources,
            additional_context=additional_context
        )
        
        return report
    
    async def _generate_report(
        self,
        topic: str,
        research_question: str,
        synthesis: SynthesizedKnowledge,
        sources: List[Dict[str, Any]],
        additional_context: Optional[str]
    ) -> ResearchReport:
        """Generate comprehensive research report using LLM"""
        
        # Prepare synthesis summary
        synthesis_summary = f"""
Key Findings: {', '.join(synthesis.key_findings[:5])}
Major Themes: {', '.join(synthesis.themes[:5])}
Consensus Points: {', '.join(synthesis.consensus_points[:3])}
Controversies: {', '.join(synthesis.controversies[:3])}
Knowledge Gaps: {', '.join(synthesis.gaps[:3])}
Recommendations: {', '.join(synthesis.recommendations[:5])}
Confidence Level: {synthesis.confidence_level}
Sources Used: {synthesis.sources_used}
"""

        prompt = f"""Generate a comprehensive research report based on the following synthesis.

Topic: {topic}
Research Question: {research_question}
Additional Context: {additional_context or "None"}

Synthesis Summary:
{synthesis_summary}

Generate a detailed research report with:
1. Executive Summary (2-3 paragraphs)
2. Detailed Analysis (comprehensive discussion of findings)
3. Methodology description
4. Limitations of the research
5. Suggestions for future research

Format as JSON:
{{
    "executive_summary": "...",
    "detailed_analysis": "...",
    "methodology": "...",
    "limitations": ["limitation1", ...],
    "future_research": ["suggestion1", ...]
}}"""

        try:
            response = await self.knowledge_synthesis.http_client.post(
                f"{self.config.llm_base_url}/api/generate",
                json={
                    "model": self.config.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=180.0
            )
            response.raise_for_status()
            data = response.json()
            report_data = json.loads(data.get("response", "{}"))
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            report_data = {
                "executive_summary": f"Research on {topic} examining: {research_question}",
                "detailed_analysis": "\n".join(synthesis.key_findings),
                "methodology": "Multi-source research using web and academic databases",
                "limitations": ["Automated synthesis may miss nuances"],
                "future_research": synthesis.gaps
            }
        
        # Compile references
        references = []
        for s in sources[:20]:
            if s.get("type") == "academic":
                ref = f"{', '.join(s.get('authors', ['Unknown'])[:3])} ({s.get('year', 'n.d.')}). {s.get('title', 'Untitled')}."
            else:
                ref = f"{s.get('title', 'Untitled')}. Retrieved from {s.get('url', 'Unknown URL')}"
            references.append(ref)
        
        return ResearchReport(
            title=f"Research Report: {topic}",
            executive_summary=report_data.get("executive_summary", ""),
            research_question=research_question,
            methodology=report_data.get("methodology", "Multi-source research methodology"),
            key_findings=synthesis.key_findings,
            detailed_analysis=report_data.get("detailed_analysis", ""),
            recommendations=synthesis.recommendations,
            limitations=report_data.get("limitations", []),
            future_research=report_data.get("future_research", synthesis.gaps),
            references=references
        )


# =============================================================================
# LangGraph-Compatible Tool Functions
# =============================================================================


@langchain_tool
def search_web(query: str, num_results: int = 10) -> str:
    """
    Search the web for information on a given query.
    
    Args:
        query: The search query string
        num_results: Maximum number of results to return (default 10)
    
    Returns:
        JSON string with search results containing titles, URLs, and snippets
    """
    async def _search():
        tool = WebSearchTool()
        try:
            results = await tool.execute(query=query, num_results=num_results)
            return json.dumps([r.model_dump() for r in results], indent=2)
        finally:
            await tool.close()
    
    return asyncio.run(_search())


@langchain_tool
def search_academic_papers(query: str, num_results: int = 20) -> str:
    """
    Search academic databases for scholarly papers and publications.
    
    Args:
        query: The search query string
        num_results: Maximum number of results to return (default 20)
    
    Returns:
        JSON string with academic papers containing titles, authors, abstracts, and citations
    """
    async def _search():
        tool = AcademicSearchTool()
        try:
            results = await tool.execute(query=query, num_results=num_results)
            return json.dumps([r.model_dump() for r in results], indent=2)
        finally:
            await tool.close()
    
    return asyncio.run(_search())


@langchain_tool
def synthesize_knowledge(topic: str, sources_json: str, research_question: str = "") -> str:
    """
    Synthesize information from multiple sources into coherent knowledge.
    
    Args:
        topic: The research topic
        sources_json: JSON string containing list of source dictionaries
        research_question: Optional specific research question
    
    Returns:
        JSON string with synthesized knowledge including findings, themes, and recommendations
    """
    async def _synthesize():
        tool = KnowledgeSynthesisTool()
        try:
            sources = json.loads(sources_json)
            result = await tool.execute(
                topic=topic,
                sources=sources,
                research_question=research_question or None
            )
            return json.dumps(result.model_dump(), indent=2)
        finally:
            await tool.close()
    
    return asyncio.run(_synthesize())


@langchain_tool
def verify_facts(claim: str, context: str = "") -> str:
    """
    Verify the accuracy of a claim or statement by cross-referencing sources.
    
    Args:
        claim: The claim or statement to verify
        context: Optional context for the claim
    
    Returns:
        JSON string with verification result including verdict, confidence, and evidence
    """
    async def _verify():
        tool = FactVerificationTool()
        try:
            result = await tool.execute(claim=claim, context=context or None)
            return json.dumps(result.model_dump(), indent=2)
        finally:
            await tool.close()
    
    return asyncio.run(_verify())


@langchain_tool
def generate_research_report(
    topic: str,
    research_question: str,
    additional_context: str = ""
) -> str:
    """
    Conduct comprehensive research and generate a detailed report.
    
    Args:
        topic: The research topic
        research_question: The specific research question to answer
        additional_context: Optional additional context or requirements
    
    Returns:
        JSON string with comprehensive research report
    """
    async def _research():
        toolkit = ResearchToolkit()
        try:
            report = await toolkit.conduct_research(
                topic=topic,
                research_question=research_question,
                additional_context=additional_context or None
            )
            return json.dumps(report.model_dump(), indent=2, default=str)
        finally:
            await toolkit.close()
    
    return asyncio.run(_research())
