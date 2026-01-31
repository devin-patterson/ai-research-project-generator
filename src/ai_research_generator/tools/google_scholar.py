"""
Google Scholar Integration Tool

Provides academic paper search via Google Scholar using the scholarly library
or SerpAPI for more reliable access.

Note: Google Scholar doesn't have an official API, so we use:
1. scholarly library (free, but rate-limited)
2. SerpAPI Google Scholar endpoint (paid, more reliable)
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field

# Try to import scholarly for direct Google Scholar access
try:
    from scholarly import scholarly, ProxyGenerator

    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False
    logger.warning("scholarly library not available. Install with: pip install scholarly")


class GoogleScholarResult(BaseModel):
    """Schema for Google Scholar search results"""

    title: str = Field(description="Paper title")
    authors: List[str] = Field(description="List of authors")
    abstract: Optional[str] = Field(default=None, description="Paper abstract")
    year: Optional[int] = Field(default=None, description="Publication year")
    citation_count: Optional[int] = Field(default=None, description="Number of citations")
    url: Optional[str] = Field(default=None, description="URL to paper")
    pdf_url: Optional[str] = Field(default=None, description="Direct PDF URL if available")
    venue: Optional[str] = Field(default=None, description="Publication venue/journal")
    scholar_id: Optional[str] = Field(default=None, description="Google Scholar paper ID")
    bibtex: Optional[str] = Field(default=None, description="BibTeX citation")


@dataclass
class GoogleScholarConfig:
    """Configuration for Google Scholar tool"""

    serpapi_key: Optional[str] = field(default_factory=lambda: os.getenv("SERPAPI_KEY"))
    use_proxy: bool = False
    proxy_type: str = "free"  # "free", "scraper", or custom
    max_results: int = 20
    timeout: float = 30.0


class GoogleScholarTool:
    """
    Tool for searching Google Scholar for academic papers.

    Supports two modes:
    1. Direct access via scholarly library (free but rate-limited)
    2. SerpAPI access (paid but more reliable)
    """

    def __init__(self, config: Optional[GoogleScholarConfig] = None):
        self.config = config or GoogleScholarConfig()
        self._http_client: Optional[httpx.AsyncClient] = None
        self._scholarly_initialized = False

    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._http_client

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _init_scholarly(self):
        """Initialize scholarly with optional proxy"""
        if not SCHOLARLY_AVAILABLE or self._scholarly_initialized:
            return

        if self.config.use_proxy:
            try:
                pg = ProxyGenerator()
                if self.config.proxy_type == "free":
                    pg.FreeProxies()
                elif self.config.proxy_type == "scraper":
                    pg.ScraperAPI(os.getenv("SCRAPER_API_KEY", ""))
                scholarly.use_proxy(pg)
                logger.info("Scholarly proxy configured")
            except Exception as e:
                logger.warning(f"Failed to configure scholarly proxy: {e}")

        self._scholarly_initialized = True

    async def search(
        self,
        query: str,
        num_results: int = 20,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        include_citations: bool = False,
    ) -> List[GoogleScholarResult]:
        """
        Search Google Scholar for papers.

        Args:
            query: Search query string
            num_results: Maximum number of results
            year_from: Filter papers from this year
            year_to: Filter papers up to this year
            include_citations: Whether to fetch citation counts (slower)

        Returns:
            List of GoogleScholarResult objects
        """
        logger.info(f"Searching Google Scholar: {query[:50]}...")

        # Try SerpAPI first if key available
        if self.config.serpapi_key:
            try:
                return await self._search_serpapi(query, num_results, year_from, year_to)
            except Exception as e:
                logger.warning(f"SerpAPI search failed: {e}")

        # Fall back to scholarly library
        if SCHOLARLY_AVAILABLE:
            try:
                return await self._search_scholarly(
                    query, num_results, year_from, year_to, include_citations
                )
            except Exception as e:
                logger.warning(f"Scholarly search failed: {e}")

        logger.error("No Google Scholar search method available")
        return []

    async def _search_serpapi(
        self, query: str, num_results: int, year_from: Optional[int], year_to: Optional[int]
    ) -> List[GoogleScholarResult]:
        """Search using SerpAPI Google Scholar endpoint"""
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.config.serpapi_key,
            "num": min(num_results, 20),  # SerpAPI max per page
        }

        if year_from:
            params["as_ylo"] = year_from
        if year_to:
            params["as_yhi"] = year_to

        response = await self.http_client.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("organic_results", []):
            # Extract authors
            authors = []
            if item.get("publication_info", {}).get("authors"):
                authors = [a.get("name", "") for a in item["publication_info"]["authors"]]
            elif item.get("publication_info", {}).get("summary"):
                # Parse from summary string
                summary = item["publication_info"]["summary"]
                if " - " in summary:
                    authors = [a.strip() for a in summary.split(" - ")[0].split(",")]

            # Extract year
            year = None
            if item.get("publication_info", {}).get("summary"):
                import re

                year_match = re.search(r"\b(19|20)\d{2}\b", item["publication_info"]["summary"])
                if year_match:
                    year = int(year_match.group())

            results.append(
                GoogleScholarResult(
                    title=item.get("title", ""),
                    authors=authors,
                    abstract=item.get("snippet"),
                    year=year,
                    citation_count=item.get("inline_links", {}).get("cited_by", {}).get("total"),
                    url=item.get("link"),
                    pdf_url=item.get("resources", [{}])[0].get("link")
                    if item.get("resources")
                    else None,
                    venue=item.get("publication_info", {}).get("summary", "").split(" - ")[-1]
                    if item.get("publication_info")
                    else None,
                    scholar_id=item.get("result_id"),
                )
            )

        return results

    async def _search_scholarly(
        self,
        query: str,
        num_results: int,
        year_from: Optional[int],
        year_to: Optional[int],
        include_citations: bool,
    ) -> List[GoogleScholarResult]:
        """Search using scholarly library (runs in thread pool)"""
        self._init_scholarly()

        def _do_search():
            results = []
            search_query = scholarly.search_pubs(query)

            for i, pub in enumerate(search_query):
                if i >= num_results:
                    break

                # Filter by year if specified
                pub_year = pub.get("bib", {}).get("pub_year")
                if pub_year:
                    try:
                        pub_year = int(pub_year)
                        if year_from and pub_year < year_from:
                            continue
                        if year_to and pub_year > year_to:
                            continue
                    except (ValueError, TypeError):
                        pub_year = None

                # Get full publication details if needed
                if include_citations:
                    try:
                        pub = scholarly.fill(pub)
                    except Exception as exc:
                        logger.warning(
                            "Failed to enrich Google Scholar publication with citations via scholarly: {error}. "
                            "Proceeding with partial data. Publication: {pub}",
                            error=exc,
                            pub=pub,
                        )
                        # Best-effort enrichment; continue without filled citation details

                bib = pub.get("bib", {})
                results.append(
                    GoogleScholarResult(
                        title=bib.get("title", ""),
                        authors=bib.get("author", [])
                        if isinstance(bib.get("author"), list)
                        else [bib.get("author", "")],
                        abstract=bib.get("abstract"),
                        year=pub_year,
                        citation_count=pub.get("num_citations"),
                        url=pub.get("pub_url") or pub.get("eprint_url"),
                        pdf_url=pub.get("eprint_url"),
                        venue=bib.get("venue") or bib.get("journal"),
                        scholar_id=pub.get("author_pub_id"),
                    )
                )

            return results

        # Run blocking scholarly calls in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _do_search)

    async def get_paper_details(self, scholar_id: str) -> Optional[GoogleScholarResult]:
        """Get detailed information about a specific paper"""
        if not SCHOLARLY_AVAILABLE:
            logger.warning("scholarly library required for paper details")
            return None

        self._init_scholarly()

        def _get_details():
            try:
                pub = scholarly.search_single_pub(scholar_id)
                if pub:
                    pub = scholarly.fill(pub)
                    bib = pub.get("bib", {})
                    return GoogleScholarResult(
                        title=bib.get("title", ""),
                        authors=bib.get("author", []),
                        abstract=bib.get("abstract"),
                        year=int(bib.get("pub_year")) if bib.get("pub_year") else None,
                        citation_count=pub.get("num_citations"),
                        url=pub.get("pub_url"),
                        pdf_url=pub.get("eprint_url"),
                        venue=bib.get("venue") or bib.get("journal"),
                        scholar_id=pub.get("author_pub_id"),
                        bibtex=scholarly.bibtex(pub) if hasattr(scholarly, "bibtex") else None,
                    )
            except Exception as e:
                logger.error(f"Failed to get paper details: {e}")
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_details)

    async def get_author_papers(
        self, author_name: str, num_results: int = 20
    ) -> List[GoogleScholarResult]:
        """Get papers by a specific author"""
        if not SCHOLARLY_AVAILABLE:
            # Fall back to regular search with author filter
            return await self.search(f"author:{author_name}", num_results)

        self._init_scholarly()

        def _get_author_papers():
            results = []
            try:
                search_query = scholarly.search_author(author_name)
                author = next(search_query, None)

                if author:
                    author = scholarly.fill(author)
                    for pub in author.get("publications", [])[:num_results]:
                        try:
                            pub = scholarly.fill(pub)
                            bib = pub.get("bib", {})
                            results.append(
                                GoogleScholarResult(
                                    title=bib.get("title", ""),
                                    authors=bib.get("author", []),
                                    abstract=bib.get("abstract"),
                                    year=int(bib.get("pub_year")) if bib.get("pub_year") else None,
                                    citation_count=pub.get("num_citations"),
                                    url=pub.get("pub_url"),
                                    venue=bib.get("venue") or bib.get("journal"),
                                )
                            )
                        except Exception:
                            continue
            except Exception as e:
                logger.error(f"Failed to get author papers: {e}")

            return results

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_author_papers)
