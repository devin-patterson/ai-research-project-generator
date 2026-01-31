#!/usr/bin/env python3
"""
Academic Search Module

Provides integration with academic search APIs for finding research papers.
Supports Semantic Scholar, OpenAlex, CrossRef, and arXiv.

Author: AI Research Assistant
Date: 2025-01-29
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

import httpx


@dataclass
class Paper:
    """Represents an academic paper"""

    title: str
    authors: List[str]
    abstract: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    citation_count: Optional[int] = None
    venue: Optional[str] = None
    source: str = "unknown"
    paper_id: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    pdf_url: Optional[str] = None

    def to_citation(self, style: str = "APA") -> str:
        """Generate citation in specified style"""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."

        if style.upper() == "APA":
            year_str = f"({self.year})" if self.year else "(n.d.)"
            venue_str = f" *{self.venue}*." if self.venue else ""
            doi_str = f" https://doi.org/{self.doi}" if self.doi else ""
            return f"{authors_str} {year_str}. {self.title}.{venue_str}{doi_str}"
        elif style.upper() == "MLA":
            year_str = str(self.year) if self.year else "n.d."
            return f'{authors_str}. "{self.title}." {self.venue or ""}, {year_str}.'
        else:
            return f"{authors_str} ({self.year}). {self.title}."


@dataclass
class SearchResult:
    """Search result containing papers and metadata"""

    papers: List[Paper]
    total_results: int
    query: str
    source: str
    page: int = 1
    per_page: int = 10
    search_time: float = 0.0


class BaseAcademicSearch(ABC):
    """Base class for academic search providers"""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0.0

    def _rate_limit(self):
        """Apply rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    @abstractmethod
    def search(self, query: str, limit: int = 10, offset: int = 0, **kwargs) -> SearchResult:
        """Search for papers"""
        pass

    @abstractmethod
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get paper details by ID"""
        pass

    def close(self):
        """Close the HTTP client"""
        self.client.close()


class SemanticScholarSearch(BaseAcademicSearch):
    """Semantic Scholar API client"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        super().__init__(api_key, timeout)
        self.rate_limit_delay = 1.0 if api_key else 3.0  # Slower without API key

    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        year_range: Optional[tuple] = None,
        fields_of_study: Optional[List[str]] = None,
        open_access_only: bool = False,
    ) -> SearchResult:
        """
        Search Semantic Scholar for papers

        Args:
            query: Search query
            limit: Number of results (max 100)
            offset: Offset for pagination
            year_range: Tuple of (start_year, end_year)
            fields_of_study: List of fields to filter by
            open_access_only: Only return open access papers
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/paper/search"

        params = {
            "query": query,
            "limit": min(limit, 100),
            "offset": offset,
            "fields": "paperId,title,abstract,authors,year,citationCount,venue,externalIds,openAccessPdf,fieldsOfStudy",
        }

        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        if open_access_only:
            params["openAccessPdf"] = ""

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        start_time = time.time()

        try:
            response = self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            papers = []
            for item in data.get("data", []):
                paper = Paper(
                    title=item.get("title", ""),
                    authors=[a.get("name", "") for a in item.get("authors", [])],
                    abstract=item.get("abstract"),
                    year=item.get("year"),
                    doi=item.get("externalIds", {}).get("DOI"),
                    citation_count=item.get("citationCount"),
                    venue=item.get("venue"),
                    source="semantic_scholar",
                    paper_id=item.get("paperId"),
                    keywords=[f.get("category", "") for f in item.get("fieldsOfStudy", []) if f],
                    pdf_url=item.get("openAccessPdf", {}).get("url")
                    if item.get("openAccessPdf")
                    else None,
                )
                papers.append(paper)

            return SearchResult(
                papers=papers,
                total_results=data.get("total", len(papers)),
                query=query,
                source="semantic_scholar",
                page=(offset // limit) + 1,
                per_page=limit,
                search_time=time.time() - start_time,
            )

        except httpx.HTTPError as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return SearchResult(
                papers=[],
                total_results=0,
                query=query,
                source="semantic_scholar",
                search_time=time.time() - start_time,
            )

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get paper details by Semantic Scholar ID"""
        self._rate_limit()

        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {
            "fields": "paperId,title,abstract,authors,year,citationCount,venue,externalIds,openAccessPdf,fieldsOfStudy,references"
        }

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            response = self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            item = response.json()

            return Paper(
                title=item.get("title", ""),
                authors=[a.get("name", "") for a in item.get("authors", [])],
                abstract=item.get("abstract"),
                year=item.get("year"),
                doi=item.get("externalIds", {}).get("DOI"),
                citation_count=item.get("citationCount"),
                venue=item.get("venue"),
                source="semantic_scholar",
                paper_id=item.get("paperId"),
                keywords=[f.get("category", "") for f in item.get("fieldsOfStudy", []) if f],
                references=[r.get("paperId", "") for r in item.get("references", []) if r],
                pdf_url=item.get("openAccessPdf", {}).get("url")
                if item.get("openAccessPdf")
                else None,
            )

        except httpx.HTTPError as e:
            logger.error(f"Semantic Scholar get paper error: {e}")
            return None

    def get_citations(self, paper_id: str, limit: int = 100) -> List[Paper]:
        """Get papers that cite this paper"""
        self._rate_limit()

        url = f"{self.BASE_URL}/paper/{paper_id}/citations"
        params = {
            "limit": min(limit, 1000),
            "fields": "paperId,title,authors,year,citationCount,venue",
        }

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            response = self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            papers = []
            for item in data.get("data", []):
                citing = item.get("citingPaper", {})
                paper = Paper(
                    title=citing.get("title", ""),
                    authors=[a.get("name", "") for a in citing.get("authors", [])],
                    year=citing.get("year"),
                    citation_count=citing.get("citationCount"),
                    venue=citing.get("venue"),
                    source="semantic_scholar",
                    paper_id=citing.get("paperId"),
                )
                papers.append(paper)

            return papers

        except httpx.HTTPError as e:
            logger.error(f"Semantic Scholar citations error: {e}")
            return []


class OpenAlexSearch(BaseAcademicSearch):
    """OpenAlex API client - Free, comprehensive academic database"""

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: Optional[str] = None, timeout: int = 30):
        super().__init__(timeout=timeout)
        self.email = email or os.getenv("OPENALEX_EMAIL")
        self.rate_limit_delay = 0.1  # OpenAlex is generous with rate limits

    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        year_range: Optional[tuple] = None,
        open_access_only: bool = False,
        sort_by: str = "relevance_score",
    ) -> SearchResult:
        """
        Search OpenAlex for papers

        Args:
            query: Search query
            limit: Number of results (max 200)
            offset: Offset for pagination
            year_range: Tuple of (start_year, end_year)
            open_access_only: Only return open access papers
            sort_by: Sort field (relevance_score, cited_by_count, publication_date)
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/works"

        # Build filter
        filters = []
        if year_range:
            filters.append(f"publication_year:{year_range[0]}-{year_range[1]}")
        if open_access_only:
            filters.append("is_oa:true")

        params = {
            "search": query,
            "per_page": min(limit, 200),
            "page": (offset // limit) + 1,
        }

        if filters:
            params["filter"] = ",".join(filters)

        if sort_by != "relevance_score":
            params["sort"] = sort_by + ":desc"

        if self.email:
            params["mailto"] = self.email

        start_time = time.time()

        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            papers = []
            for item in data.get("results", []):
                # Extract authors
                authors = []
                for authorship in item.get("authorships", []):
                    author = authorship.get("author", {})
                    if author.get("display_name"):
                        authors.append(author["display_name"])

                # Get best open access URL
                pdf_url = None
                for location in item.get("locations", []):
                    if location.get("pdf_url"):
                        pdf_url = location["pdf_url"]
                        break

                # Safely get venue
                venue = None
                primary_loc = item.get("primary_location")
                if primary_loc:
                    source = primary_loc.get("source")
                    if source:
                        venue = source.get("display_name")

                paper = Paper(
                    title=item.get("title", ""),
                    authors=authors,
                    abstract=item.get("abstract_inverted_index"),  # Needs processing
                    year=item.get("publication_year"),
                    doi=item.get("doi", "").replace("https://doi.org/", "")
                    if item.get("doi")
                    else None,
                    url=item.get("doi"),
                    citation_count=item.get("cited_by_count"),
                    venue=venue,
                    source="openalex",
                    paper_id=item.get("id", "").replace("https://openalex.org/", ""),
                    keywords=[c.get("display_name", "") for c in item.get("concepts", [])[:5]],
                    pdf_url=pdf_url,
                )

                # Process inverted index abstract if present
                if isinstance(paper.abstract, dict):
                    paper.abstract = self._reconstruct_abstract(paper.abstract)

                papers.append(paper)

            return SearchResult(
                papers=papers,
                total_results=data.get("meta", {}).get("count", len(papers)),
                query=query,
                source="openalex",
                page=(offset // limit) + 1,
                per_page=limit,
                search_time=time.time() - start_time,
            )

        except httpx.HTTPError as e:
            logger.error(f"OpenAlex search error: {e}")
            return SearchResult(
                papers=[],
                total_results=0,
                query=query,
                source="openalex",
                search_time=time.time() - start_time,
            )

    def _reconstruct_abstract(self, inverted_index: Dict[str, List[int]]) -> str:
        """Reconstruct abstract from OpenAlex inverted index format"""
        if not inverted_index:
            return ""

        # Find max position
        max_pos = 0
        for positions in inverted_index.values():
            if positions:
                max_pos = max(max_pos, max(positions))

        # Reconstruct
        words = [""] * (max_pos + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word

        return " ".join(words)

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get paper details by OpenAlex ID"""
        self._rate_limit()

        # Handle both full URL and just ID
        if not paper_id.startswith("https://"):
            paper_id = f"https://openalex.org/{paper_id}"

        url = paper_id

        params = {}
        if self.email:
            params["mailto"] = self.email

        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            item = response.json()

            authors = []
            for authorship in item.get("authorships", []):
                author = authorship.get("author", {})
                if author.get("display_name"):
                    authors.append(author["display_name"])

            pdf_url = None
            for location in item.get("locations", []):
                if location.get("pdf_url"):
                    pdf_url = location["pdf_url"]
                    break

            abstract = item.get("abstract_inverted_index")
            if isinstance(abstract, dict):
                abstract = self._reconstruct_abstract(abstract)

            return Paper(
                title=item.get("title", ""),
                authors=authors,
                abstract=abstract,
                year=item.get("publication_year"),
                doi=item.get("doi", "").replace("https://doi.org/", "")
                if item.get("doi")
                else None,
                url=item.get("doi"),
                citation_count=item.get("cited_by_count"),
                venue=item.get("primary_location", {}).get("source", {}).get("display_name"),
                source="openalex",
                paper_id=item.get("id", "").replace("https://openalex.org/", ""),
                keywords=[c.get("display_name", "") for c in item.get("concepts", [])[:10]],
                pdf_url=pdf_url,
            )

        except httpx.HTTPError as e:
            logger.error(f"OpenAlex get paper error: {e}")
            return None


class CrossRefSearch(BaseAcademicSearch):
    """CrossRef API client - DOI and metadata"""

    BASE_URL = "https://api.crossref.org"

    def __init__(self, email: Optional[str] = None, timeout: int = 30):
        super().__init__(timeout=timeout)
        self.email = email or os.getenv("CROSSREF_EMAIL")
        self.rate_limit_delay = 0.5

    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        year_range: Optional[tuple] = None,
        sort_by: str = "relevance",
    ) -> SearchResult:
        """
        Search CrossRef for papers

        Args:
            query: Search query
            limit: Number of results (max 1000)
            offset: Offset for pagination
            year_range: Tuple of (start_year, end_year)
            sort_by: Sort field (relevance, published, is-referenced-by-count)
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/works"

        params = {
            "query": query,
            "rows": min(limit, 1000),
            "offset": offset,
            "sort": sort_by,
            "order": "desc",
        }

        if year_range:
            params["filter"] = f"from-pub-date:{year_range[0]},until-pub-date:{year_range[1]}"

        headers = {}
        if self.email:
            headers["User-Agent"] = f"ResearchProjectGenerator/1.0 (mailto:{self.email})"

        start_time = time.time()

        try:
            response = self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            papers = []
            for item in data.get("message", {}).get("items", []):
                # Extract authors
                authors = []
                for author in item.get("author", []):
                    name_parts = []
                    if author.get("given"):
                        name_parts.append(author["given"])
                    if author.get("family"):
                        name_parts.append(author["family"])
                    if name_parts:
                        authors.append(" ".join(name_parts))

                # Get year from published date
                year = None
                published = item.get("published-print") or item.get("published-online")
                if published and published.get("date-parts"):
                    date_parts = published["date-parts"][0]
                    if date_parts:
                        year = date_parts[0]

                paper = Paper(
                    title=item.get("title", [""])[0] if item.get("title") else "",
                    authors=authors,
                    abstract=item.get("abstract", "")
                    .replace("<jats:p>", "")
                    .replace("</jats:p>", ""),
                    year=year,
                    doi=item.get("DOI"),
                    url=item.get("URL"),
                    citation_count=item.get("is-referenced-by-count"),
                    venue=item.get("container-title", [""])[0]
                    if item.get("container-title")
                    else None,
                    source="crossref",
                    paper_id=item.get("DOI"),
                )
                papers.append(paper)

            return SearchResult(
                papers=papers,
                total_results=data.get("message", {}).get("total-results", len(papers)),
                query=query,
                source="crossref",
                page=(offset // limit) + 1,
                per_page=limit,
                search_time=time.time() - start_time,
            )

        except httpx.HTTPError as e:
            logger.error(f"CrossRef search error: {e}")
            return SearchResult(
                papers=[],
                total_results=0,
                query=query,
                source="crossref",
                search_time=time.time() - start_time,
            )

    def get_paper(self, doi: str) -> Optional[Paper]:
        """Get paper details by DOI"""
        self._rate_limit()

        # Clean DOI
        doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

        url = f"{self.BASE_URL}/works/{doi}"

        headers = {}
        if self.email:
            headers["User-Agent"] = f"ResearchProjectGenerator/1.0 (mailto:{self.email})"

        try:
            response = self.client.get(url, headers=headers)
            response.raise_for_status()
            item = response.json().get("message", {})

            authors = []
            for author in item.get("author", []):
                name_parts = []
                if author.get("given"):
                    name_parts.append(author["given"])
                if author.get("family"):
                    name_parts.append(author["family"])
                if name_parts:
                    authors.append(" ".join(name_parts))

            year = None
            published = item.get("published-print") or item.get("published-online")
            if published and published.get("date-parts"):
                date_parts = published["date-parts"][0]
                if date_parts:
                    year = date_parts[0]

            return Paper(
                title=item.get("title", [""])[0] if item.get("title") else "",
                authors=authors,
                abstract=item.get("abstract", "").replace("<jats:p>", "").replace("</jats:p>", ""),
                year=year,
                doi=item.get("DOI"),
                url=item.get("URL"),
                citation_count=item.get("is-referenced-by-count"),
                venue=item.get("container-title", [""])[0] if item.get("container-title") else None,
                source="crossref",
                paper_id=item.get("DOI"),
                references=[r.get("DOI", "") for r in item.get("reference", []) if r.get("DOI")],
            )

        except httpx.HTTPError as e:
            logger.error(f"CrossRef get paper error: {e}")
            return None


class ArxivSearch(BaseAcademicSearch):
    """arXiv API client - Preprints in physics, math, CS, etc."""

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, timeout: int = 30):
        super().__init__(timeout=timeout)
        self.rate_limit_delay = 3.0  # arXiv requires 3 second delay

    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        categories: Optional[List[str]] = None,
        sort_by: str = "relevance",
    ) -> SearchResult:
        """
        Search arXiv for papers

        Args:
            query: Search query
            limit: Number of results (max 100)
            offset: Offset for pagination
            categories: List of arXiv categories (e.g., ['cs.AI', 'cs.CL'])
            sort_by: Sort field (relevance, lastUpdatedDate, submittedDate)
        """
        self._rate_limit()

        # Build query
        search_query = f"all:{query}"
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({search_query}) AND ({cat_query})"

        params = {
            "search_query": search_query,
            "start": offset,
            "max_results": min(limit, 100),
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        start_time = time.time()

        try:
            response = self.client.get(self.BASE_URL, params=params)
            response.raise_for_status()

            # Parse XML response
            import xml.etree.ElementTree as ET  # nosec: B405

            root = ET.fromstring(response.text)  # nosec: B314

            # Define namespaces
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

            papers = []
            for entry in root.findall("atom:entry", ns):
                # Extract authors
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None and name.text:
                        authors.append(name.text)

                # Extract year from published date
                published = entry.find("atom:published", ns)
                year = None
                if published is not None and published.text:
                    year = int(published.text[:4])

                # Extract arXiv ID
                id_elem = entry.find("atom:id", ns)
                arxiv_id = ""
                if id_elem is not None and id_elem.text:
                    arxiv_id = id_elem.text.split("/abs/")[-1]

                # Get PDF link
                pdf_url = None
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href")
                        break

                # Get categories
                categories_found = []
                for cat in entry.findall("arxiv:primary_category", ns):
                    if cat.get("term"):
                        categories_found.append(cat.get("term"))
                for cat in entry.findall("atom:category", ns):
                    if cat.get("term") and cat.get("term") not in categories_found:
                        categories_found.append(cat.get("term"))

                title_elem = entry.find("atom:title", ns)
                abstract_elem = entry.find("atom:summary", ns)

                paper = Paper(
                    title=title_elem.text.strip().replace("\n", " ")
                    if title_elem is not None
                    else "",
                    authors=authors,
                    abstract=abstract_elem.text.strip().replace("\n", " ")
                    if abstract_elem is not None
                    else None,
                    year=year,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    source="arxiv",
                    paper_id=arxiv_id,
                    keywords=categories_found[:5],
                    pdf_url=pdf_url,
                )
                papers.append(paper)

            # Get total results from opensearch
            total_elem = root.find(".//{http://a9.com/-/spec/opensearch/1.1/}totalResults")
            total = int(total_elem.text) if total_elem is not None else len(papers)

            return SearchResult(
                papers=papers,
                total_results=total,
                query=query,
                source="arxiv",
                page=(offset // limit) + 1,
                per_page=limit,
                search_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return SearchResult(
                papers=[],
                total_results=0,
                query=query,
                source="arxiv",
                search_time=time.time() - start_time,
            )

    def get_paper(self, arxiv_id: str) -> Optional[Paper]:
        """Get paper details by arXiv ID"""
        # Clean ID
        arxiv_id = arxiv_id.replace("https://arxiv.org/abs/", "").replace("arXiv:", "")

        result = self.search(f"id:{arxiv_id}", limit=1)
        if result.papers:
            return result.papers[0]
        return None


class UnifiedAcademicSearch:
    """Unified interface for searching across multiple academic databases"""

    def __init__(
        self,
        semantic_scholar_key: Optional[str] = None,
        openalex_email: Optional[str] = None,
        crossref_email: Optional[str] = None,
    ):
        """
        Initialize unified search with optional API keys/emails

        Args:
            semantic_scholar_key: Semantic Scholar API key (optional, increases rate limit)
            openalex_email: Email for OpenAlex polite pool (recommended)
            crossref_email: Email for CrossRef polite pool (recommended)
        """
        self.semantic_scholar = SemanticScholarSearch(api_key=semantic_scholar_key)
        self.openalex = OpenAlexSearch(email=openalex_email)
        self.crossref = CrossRefSearch(email=crossref_email)
        self.arxiv = ArxivSearch()

    def search_all(
        self,
        query: str,
        limit_per_source: int = 10,
        year_range: Optional[tuple] = None,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, SearchResult]:
        """
        Search across all sources

        Args:
            query: Search query
            limit_per_source: Number of results per source
            year_range: Tuple of (start_year, end_year)
            sources: List of sources to search (default: all)

        Returns:
            Dictionary mapping source name to SearchResult
        """
        if sources is None:
            sources = ["semantic_scholar", "openalex", "crossref", "arxiv"]

        results = {}

        if "semantic_scholar" in sources:
            logger.info("Searching Semantic Scholar...")
            results["semantic_scholar"] = self.semantic_scholar.search(
                query, limit=limit_per_source, year_range=year_range
            )

        if "openalex" in sources:
            logger.info("Searching OpenAlex...")
            results["openalex"] = self.openalex.search(
                query, limit=limit_per_source, year_range=year_range
            )

        if "crossref" in sources:
            logger.info("Searching CrossRef...")
            results["crossref"] = self.crossref.search(
                query, limit=limit_per_source, year_range=year_range
            )

        if "arxiv" in sources:
            logger.info("Searching arXiv...")
            results["arxiv"] = self.arxiv.search(query, limit=limit_per_source)

        return results

    def search_merged(
        self,
        query: str,
        limit: int = 20,
        year_range: Optional[tuple] = None,
        sources: Optional[List[str]] = None,
        deduplicate: bool = True,
    ) -> List[Paper]:
        """
        Search all sources and merge results

        Args:
            query: Search query
            limit: Total number of results to return
            year_range: Tuple of (start_year, end_year)
            sources: List of sources to search
            deduplicate: Remove duplicate papers based on DOI/title

        Returns:
            List of papers sorted by citation count
        """
        # Get results from all sources
        all_results = self.search_all(
            query,
            limit_per_source=limit // 2,  # Get more from each to allow for deduplication
            year_range=year_range,
            sources=sources,
        )

        # Merge all papers
        all_papers = []
        for result in all_results.values():
            all_papers.extend(result.papers)

        # Deduplicate
        if deduplicate:
            seen_dois = set()
            seen_titles = set()
            unique_papers = []

            for paper in all_papers:
                # Check DOI
                if paper.doi:
                    if paper.doi.lower() in seen_dois:
                        continue
                    seen_dois.add(paper.doi.lower())

                # Check title similarity (simple exact match)
                title_key = paper.title.lower().strip()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

                unique_papers.append(paper)

            all_papers = unique_papers

        # Sort by citation count (descending)
        all_papers.sort(key=lambda p: p.citation_count or 0, reverse=True)

        return all_papers[:limit]

    def close(self):
        """Close all clients"""
        self.semantic_scholar.close()
        self.openalex.close()
        self.crossref.close()
        self.arxiv.close()


def main():
    """Example usage of academic search"""
    search = UnifiedAcademicSearch()

    query = "remote work productivity employee well-being"
    print(f"Searching for: {query}\n")

    # Search all sources
    results = search.search_all(query, limit_per_source=5, year_range=(2020, 2025))

    for source, result in results.items():
        print(
            f"\n=== {source.upper()} ({result.total_results} total, {len(result.papers)} returned) ==="
        )
        for paper in result.papers[:3]:
            print(f"\nTitle: {paper.title}")
            print(f"Authors: {', '.join(paper.authors[:3])}")
            print(f"Year: {paper.year}")
            print(f"Citations: {paper.citation_count}")
            if paper.doi:
                print(f"DOI: {paper.doi}")

    # Get merged results
    print("\n\n=== MERGED RESULTS (Top 10 by citations) ===")
    merged = search.search_merged(query, limit=10, year_range=(2020, 2025))

    for i, paper in enumerate(merged, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   Source: {paper.source} | Year: {paper.year} | Citations: {paper.citation_count}")
        print(f"   {paper.to_citation('APA')}")

    search.close()


if __name__ == "__main__":
    main()
