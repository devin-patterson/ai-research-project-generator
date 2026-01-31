"""
Citation Management Module

Provides citation formatting in multiple styles (BibTeX, APA, MLA, Chicago, IEEE, Harvard)
and citation management utilities for research projects.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field


class CitationStyle(str, Enum):
    """Supported citation styles"""

    BIBTEX = "bibtex"
    APA = "apa"
    APA7 = "apa7"
    MLA = "mla"
    MLA9 = "mla9"
    CHICAGO = "chicago"
    CHICAGO_AUTHOR_DATE = "chicago_author_date"
    IEEE = "ieee"
    HARVARD = "harvard"
    VANCOUVER = "vancouver"


class PublicationType(str, Enum):
    """Types of publications for citation formatting"""

    ARTICLE = "article"
    BOOK = "book"
    INPROCEEDINGS = "inproceedings"
    CONFERENCE = "conference"
    THESIS = "thesis"
    REPORT = "report"
    WEBPAGE = "webpage"
    MISC = "misc"


class Citation(BaseModel):
    """Structured citation data"""

    # Required fields
    title: str = Field(description="Title of the work")
    authors: List[str] = Field(description="List of authors")
    year: Optional[int] = Field(default=None, description="Publication year")

    # Publication details
    publication_type: PublicationType = Field(default=PublicationType.ARTICLE)
    journal: Optional[str] = Field(default=None, description="Journal name")
    volume: Optional[str] = Field(default=None, description="Volume number")
    issue: Optional[str] = Field(default=None, description="Issue number")
    pages: Optional[str] = Field(default=None, description="Page range")

    # Book-specific
    publisher: Optional[str] = Field(default=None, description="Publisher name")
    edition: Optional[str] = Field(default=None, description="Edition")
    editors: Optional[List[str]] = Field(default=None, description="Editors")

    # Conference-specific
    booktitle: Optional[str] = Field(default=None, description="Conference/book title")
    location: Optional[str] = Field(default=None, description="Conference location")

    # Digital identifiers
    doi: Optional[str] = Field(default=None, description="DOI")
    url: Optional[str] = Field(default=None, description="URL")
    isbn: Optional[str] = Field(default=None, description="ISBN")
    issn: Optional[str] = Field(default=None, description="ISSN")

    # Access information
    accessed_date: Optional[str] = Field(default=None, description="Date accessed for web sources")

    # Internal tracking
    citation_key: Optional[str] = Field(default=None, description="Unique citation key")

    def generate_key(self) -> str:
        """Generate a unique citation key"""
        if self.citation_key:
            return self.citation_key

        # Generate from first author's last name and year
        if self.authors:
            first_author = self.authors[0]
            # Extract last name
            if "," in first_author:
                last_name = first_author.split(",")[0].strip()
            else:
                parts = first_author.split()
                last_name = parts[-1] if parts else "unknown"

            # Clean the last name
            last_name = re.sub(r"[^a-zA-Z]", "", last_name).lower()
        else:
            last_name = "unknown"

        year_str = str(self.year) if self.year else "nd"

        # Add first word of title for uniqueness
        title_word = ""
        if self.title:
            words = re.findall(r"\b[a-zA-Z]+\b", self.title)
            if words:
                title_word = words[0].lower()

        self.citation_key = f"{last_name}{year_str}{title_word}"
        return self.citation_key


class CitationFormatter:
    """Formats citations in various academic styles"""

    @staticmethod
    def format(citation: Citation, style: CitationStyle) -> str:
        """
        Format a citation in the specified style.

        Args:
            citation: Citation object with publication details
            style: Citation style to use

        Returns:
            Formatted citation string
        """
        formatters = {
            CitationStyle.BIBTEX: CitationFormatter._format_bibtex,
            CitationStyle.APA: CitationFormatter._format_apa,
            CitationStyle.APA7: CitationFormatter._format_apa7,
            CitationStyle.MLA: CitationFormatter._format_mla,
            CitationStyle.MLA9: CitationFormatter._format_mla9,
            CitationStyle.CHICAGO: CitationFormatter._format_chicago,
            CitationStyle.CHICAGO_AUTHOR_DATE: CitationFormatter._format_chicago_author_date,
            CitationStyle.IEEE: CitationFormatter._format_ieee,
            CitationStyle.HARVARD: CitationFormatter._format_harvard,
            CitationStyle.VANCOUVER: CitationFormatter._format_vancouver,
        }

        formatter = formatters.get(style, CitationFormatter._format_apa)
        return formatter(citation)

    @staticmethod
    def _format_authors_apa(authors: List[str], max_authors: int = 20) -> str:
        """Format authors in APA style"""
        if not authors:
            return ""

        formatted = []
        for author in authors[:max_authors]:
            if "," in author:
                # Already in "Last, First" format
                parts = author.split(",", 1)
                last = parts[0].strip()
                first = parts[1].strip() if len(parts) > 1 else ""
                initials = (
                    ". ".join([n[0].upper() for n in first.split() if n]) + "." if first else ""
                )
                formatted.append(f"{last}, {initials}" if initials else last)
            else:
                # "First Last" format
                parts = author.split()
                if len(parts) >= 2:
                    last = parts[-1]
                    initials = ". ".join([n[0].upper() for n in parts[:-1]]) + "."
                    formatted.append(f"{last}, {initials}")
                else:
                    formatted.append(author)

        if len(formatted) == 1:
            return formatted[0]
        elif len(formatted) == 2:
            return f"{formatted[0]} & {formatted[1]}"
        elif len(authors) > max_authors:
            return f"{', '.join(formatted[:19])}, ... {formatted[-1]}"
        else:
            return f"{', '.join(formatted[:-1])}, & {formatted[-1]}"

    @staticmethod
    def _format_authors_mla(authors: List[str]) -> str:
        """Format authors in MLA style"""
        if not authors:
            return ""

        if len(authors) == 1:
            author = authors[0]
            if "," in author:
                return author
            parts = author.split()
            if len(parts) >= 2:
                return f"{parts[-1]}, {' '.join(parts[:-1])}"
            return author
        elif len(authors) == 2:
            first = authors[0]
            if "," not in first:
                parts = first.split()
                if len(parts) >= 2:
                    first = f"{parts[-1]}, {' '.join(parts[:-1])}"
            return f"{first}, and {authors[1]}"
        else:
            first = authors[0]
            if "," not in first:
                parts = first.split()
                if len(parts) >= 2:
                    first = f"{parts[-1]}, {' '.join(parts[:-1])}"
            return f"{first}, et al."

    @staticmethod
    def _format_bibtex(citation: Citation) -> str:
        """Format citation as BibTeX entry"""
        key = citation.generate_key()
        entry_type = citation.publication_type.value

        lines = [f"@{entry_type}{{{key},"]

        # Required fields
        lines.append(f"  title = {{{citation.title}}},")

        if citation.authors:
            authors_str = " and ".join(citation.authors)
            lines.append(f"  author = {{{authors_str}}},")

        if citation.year:
            lines.append(f"  year = {{{citation.year}}},")

        # Optional fields based on type
        if citation.journal:
            lines.append(f"  journal = {{{citation.journal}}},")

        if citation.booktitle:
            lines.append(f"  booktitle = {{{citation.booktitle}}},")

        if citation.volume:
            lines.append(f"  volume = {{{citation.volume}}},")

        if citation.issue:
            lines.append(f"  number = {{{citation.issue}}},")

        if citation.pages:
            lines.append(f"  pages = {{{citation.pages}}},")

        if citation.publisher:
            lines.append(f"  publisher = {{{citation.publisher}}},")

        if citation.doi:
            lines.append(f"  doi = {{{citation.doi}}},")

        if citation.url:
            lines.append(f"  url = {{{citation.url}}},")

        if citation.isbn:
            lines.append(f"  isbn = {{{citation.isbn}}},")

        lines.append("}")

        return "\n".join(lines)

    @staticmethod
    def _format_apa(citation: Citation) -> str:
        """Format citation in APA 6th edition style"""
        parts = []

        # Authors
        if citation.authors:
            parts.append(CitationFormatter._format_authors_apa(citation.authors))

        # Year
        if citation.year:
            parts.append(f"({citation.year}).")
        else:
            parts.append("(n.d.).")

        # Title
        if citation.title:
            if citation.publication_type in [
                PublicationType.BOOK,
                PublicationType.REPORT,
                PublicationType.THESIS,
            ]:
                parts.append(f"*{citation.title}*.")
            else:
                parts.append(f"{citation.title}.")

        # Journal/Source
        if citation.journal:
            journal_part = f"*{citation.journal}*"
            if citation.volume:
                journal_part += f", *{citation.volume}*"
                if citation.issue:
                    journal_part += f"({citation.issue})"
            if citation.pages:
                journal_part += f", {citation.pages}"
            journal_part += "."
            parts.append(journal_part)
        elif citation.booktitle:
            parts.append(f"In *{citation.booktitle}*.")
        elif citation.publisher:
            parts.append(f"{citation.publisher}.")

        # DOI or URL
        if citation.doi:
            parts.append(f"https://doi.org/{citation.doi}")
        elif citation.url:
            parts.append(f"Retrieved from {citation.url}")

        return " ".join(parts)

    @staticmethod
    def _format_apa7(citation: Citation) -> str:
        """Format citation in APA 7th edition style"""
        parts = []

        # Authors (APA 7 shows up to 20 authors)
        if citation.authors:
            parts.append(CitationFormatter._format_authors_apa(citation.authors, max_authors=20))

        # Year
        if citation.year:
            parts.append(f"({citation.year}).")
        else:
            parts.append("(n.d.).")

        # Title
        if citation.title:
            if citation.publication_type in [
                PublicationType.BOOK,
                PublicationType.REPORT,
                PublicationType.THESIS,
            ]:
                parts.append(f"*{citation.title}*.")
            else:
                parts.append(f"{citation.title}.")

        # Journal/Source
        if citation.journal:
            journal_part = f"*{citation.journal}*"
            if citation.volume:
                journal_part += f", *{citation.volume}*"
                if citation.issue:
                    journal_part += f"({citation.issue})"
            if citation.pages:
                journal_part += f", {citation.pages}"
            journal_part += "."
            parts.append(journal_part)
        elif citation.booktitle:
            parts.append(f"In *{citation.booktitle}*.")
        elif citation.publisher:
            parts.append(f"{citation.publisher}.")

        # DOI (APA 7 uses https://doi.org/ format)
        if citation.doi:
            doi = citation.doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
            parts.append(f"https://doi.org/{doi}")
        elif citation.url:
            parts.append(citation.url)

        return " ".join(parts)

    @staticmethod
    def _format_mla(citation: Citation) -> str:
        """Format citation in MLA 8th edition style"""
        parts = []

        # Authors
        if citation.authors:
            parts.append(CitationFormatter._format_authors_mla(citation.authors) + ".")

        # Title
        if citation.title:
            if citation.publication_type in [PublicationType.BOOK]:
                parts.append(f"*{citation.title}*.")
            else:
                parts.append(f'"{citation.title}."')

        # Container (journal, book, etc.)
        if citation.journal:
            container = f"*{citation.journal}*"
            if citation.volume:
                container += f", vol. {citation.volume}"
            if citation.issue:
                container += f", no. {citation.issue}"
            if citation.year:
                container += f", {citation.year}"
            if citation.pages:
                container += f", pp. {citation.pages}"
            container += "."
            parts.append(container)
        elif citation.booktitle:
            parts.append(f"*{citation.booktitle}*,")
            if citation.publisher:
                parts.append(f"{citation.publisher},")
            if citation.year:
                parts.append(f"{citation.year}.")
        elif citation.publisher:
            parts.append(f"{citation.publisher},")
            if citation.year:
                parts.append(f"{citation.year}.")

        # DOI or URL
        if citation.doi:
            parts.append(f"doi:{citation.doi}.")
        elif citation.url:
            parts.append(f"{citation.url}.")
            if citation.accessed_date:
                parts.append(f"Accessed {citation.accessed_date}.")

        return " ".join(parts)

    @staticmethod
    def _format_mla9(citation: Citation) -> str:
        """Format citation in MLA 9th edition style"""
        # MLA 9 is similar to MLA 8 with minor updates
        return CitationFormatter._format_mla(citation)

    @staticmethod
    def _format_chicago(citation: Citation) -> str:
        """Format citation in Chicago Manual of Style (Notes-Bibliography)"""
        parts = []

        # Authors (full names, not inverted for first author in notes)
        if citation.authors:
            if len(citation.authors) == 1:
                parts.append(f"{citation.authors[0]},")
            elif len(citation.authors) == 2:
                parts.append(f"{citation.authors[0]} and {citation.authors[1]},")
            else:
                parts.append(f"{citation.authors[0]} et al.,")

        # Title
        if citation.title:
            if citation.publication_type in [PublicationType.BOOK]:
                parts.append(f"*{citation.title}*")
            else:
                parts.append(f'"{citation.title},"')

        # Journal/Source
        if citation.journal:
            journal_part = f"*{citation.journal}*"
            if citation.volume:
                journal_part += f" {citation.volume}"
                if citation.issue:
                    journal_part += f", no. {citation.issue}"
            if citation.year:
                journal_part += f" ({citation.year})"
            if citation.pages:
                journal_part += f": {citation.pages}"
            parts.append(journal_part + ".")
        elif citation.publisher and citation.year:
            parts.append(f"({citation.publisher}, {citation.year}).")

        # DOI
        if citation.doi:
            parts.append(f"https://doi.org/{citation.doi}.")

        return " ".join(parts)

    @staticmethod
    def _format_chicago_author_date(citation: Citation) -> str:
        """Format citation in Chicago Author-Date style"""
        parts = []

        # Authors
        if citation.authors:
            formatted_authors = []
            for i, author in enumerate(citation.authors[:3]):
                if "," in author:
                    formatted_authors.append(author)
                else:
                    author_parts = author.split()
                    if len(author_parts) >= 2:
                        formatted_authors.append(
                            f"{author_parts[-1]}, {' '.join(author_parts[:-1])}"
                        )
                    else:
                        formatted_authors.append(author)

            if len(citation.authors) > 3:
                parts.append(f"{formatted_authors[0]}, et al.")
            elif len(formatted_authors) == 2:
                parts.append(f"{formatted_authors[0]}, and {formatted_authors[1]}.")
            else:
                parts.append(f"{', '.join(formatted_authors)}.")

        # Year
        if citation.year:
            parts.append(f"{citation.year}.")

        # Title
        if citation.title:
            if citation.publication_type in [PublicationType.BOOK]:
                parts.append(f"*{citation.title}*.")
            else:
                parts.append(f'"{citation.title}."')

        # Journal
        if citation.journal:
            journal_part = f"*{citation.journal}*"
            if citation.volume:
                journal_part += f" {citation.volume}"
                if citation.issue:
                    journal_part += f" ({citation.issue})"
            if citation.pages:
                journal_part += f": {citation.pages}"
            parts.append(journal_part + ".")

        # DOI
        if citation.doi:
            parts.append(f"https://doi.org/{citation.doi}.")

        return " ".join(parts)

    @staticmethod
    def _format_ieee(citation: Citation) -> str:
        """Format citation in IEEE style"""
        parts = []

        # Authors (initials first)
        if citation.authors:
            formatted = []
            for author in citation.authors[:6]:
                if "," in author:
                    last, first = author.split(",", 1)
                    initials = ". ".join([n[0].upper() for n in first.strip().split() if n]) + "."
                    formatted.append(f"{initials} {last.strip()}")
                else:
                    author_parts = author.split()
                    if len(author_parts) >= 2:
                        initials = ". ".join([n[0].upper() for n in author_parts[:-1]]) + "."
                        formatted.append(f"{initials} {author_parts[-1]}")
                    else:
                        formatted.append(author)

            if len(citation.authors) > 6:
                parts.append(f"{', '.join(formatted)}, et al.,")
            else:
                parts.append(f"{', '.join(formatted)},")

        # Title
        if citation.title:
            parts.append(f'"{citation.title},"')

        # Journal/Conference
        if citation.journal:
            parts.append(f"*{citation.journal}*,")
            if citation.volume:
                parts.append(f"vol. {citation.volume},")
            if citation.issue:
                parts.append(f"no. {citation.issue},")
            if citation.pages:
                parts.append(f"pp. {citation.pages},")
        elif citation.booktitle:
            parts.append(f"in *{citation.booktitle}*,")

        # Year
        if citation.year:
            parts.append(f"{citation.year}.")

        # DOI
        if citation.doi:
            parts.append(f"doi: {citation.doi}.")

        return " ".join(parts)

    @staticmethod
    def _format_harvard(citation: Citation) -> str:
        """Format citation in Harvard style"""
        parts = []

        # Authors
        if citation.authors:
            formatted = []
            for author in citation.authors[:3]:
                if "," in author:
                    formatted.append(author.split(",")[0].strip())
                else:
                    formatted.append(author.split()[-1])

            if len(citation.authors) > 3:
                parts.append(f"{formatted[0]} et al.")
            elif len(formatted) == 2:
                parts.append(f"{formatted[0]} and {formatted[1]}")
            else:
                parts.append(", ".join(formatted))

        # Year
        if citation.year:
            parts.append(f"({citation.year})")
        else:
            parts.append("(n.d.)")

        # Title
        if citation.title:
            if citation.publication_type in [PublicationType.BOOK]:
                parts.append(f"*{citation.title}*.")
            else:
                parts.append(f"'{citation.title}',")

        # Journal
        if citation.journal:
            journal_part = f"*{citation.journal}*"
            if citation.volume:
                journal_part += f", {citation.volume}"
                if citation.issue:
                    journal_part += f"({citation.issue})"
            if citation.pages:
                journal_part += f", pp. {citation.pages}"
            parts.append(journal_part + ".")
        elif citation.publisher:
            parts.append(f"{citation.publisher}.")

        # DOI
        if citation.doi:
            parts.append(f"doi: {citation.doi}")

        return " ".join(parts)

    @staticmethod
    def _format_vancouver(citation: Citation) -> str:
        """Format citation in Vancouver style (used in medical journals)"""
        parts = []

        # Authors (last name + initials, no periods)
        if citation.authors:
            formatted = []
            for author in citation.authors[:6]:
                if "," in author:
                    last, first = author.split(",", 1)
                    initials = "".join([n[0].upper() for n in first.strip().split() if n])
                    formatted.append(f"{last.strip()} {initials}")
                else:
                    author_parts = author.split()
                    if len(author_parts) >= 2:
                        initials = "".join([n[0].upper() for n in author_parts[:-1]])
                        formatted.append(f"{author_parts[-1]} {initials}")
                    else:
                        formatted.append(author)

            if len(citation.authors) > 6:
                parts.append(f"{', '.join(formatted)}, et al.")
            else:
                parts.append(f"{', '.join(formatted)}.")

        # Title
        if citation.title:
            parts.append(f"{citation.title}.")

        # Journal
        if citation.journal:
            journal_part = citation.journal
            if citation.year:
                journal_part += f". {citation.year}"
            if citation.volume:
                journal_part += f";{citation.volume}"
                if citation.issue:
                    journal_part += f"({citation.issue})"
            if citation.pages:
                journal_part += f":{citation.pages}"
            parts.append(journal_part + ".")

        # DOI
        if citation.doi:
            parts.append(f"doi: {citation.doi}")

        return " ".join(parts)


class CitationManager:
    """
    Manages a collection of citations for a research project.

    Provides functionality for:
    - Adding and organizing citations
    - Generating bibliographies in various styles
    - Exporting to BibTeX files
    - In-text citation generation
    """

    def __init__(self, default_style: CitationStyle = CitationStyle.APA7):
        self.citations: Dict[str, Citation] = {}
        self.default_style = default_style
        self._citation_order: List[str] = []  # For numbered styles

    def add_citation(self, citation: Citation) -> str:
        """
        Add a citation to the manager.

        Args:
            citation: Citation object to add

        Returns:
            Citation key
        """
        key = citation.generate_key()

        # Handle duplicate keys
        if key in self.citations:
            base_key = key
            counter = 1
            while key in self.citations:
                key = f"{base_key}{chr(ord('a') + counter)}"
                counter += 1
            citation.citation_key = key

        self.citations[key] = citation
        self._citation_order.append(key)
        logger.debug(f"Added citation: {key}")

        return key

    def add_from_dict(self, data: Dict[str, Any]) -> str:
        """Add a citation from a dictionary"""
        citation = Citation(**data)
        return self.add_citation(citation)

    def get_citation(self, key: str) -> Optional[Citation]:
        """Get a citation by key"""
        return self.citations.get(key)

    def remove_citation(self, key: str) -> bool:
        """Remove a citation by key"""
        if key in self.citations:
            del self.citations[key]
            self._citation_order.remove(key)
            return True
        return False

    def format_citation(self, key: str, style: Optional[CitationStyle] = None) -> Optional[str]:
        """Format a single citation"""
        citation = self.citations.get(key)
        if not citation:
            return None

        return CitationFormatter.format(citation, style or self.default_style)

    def generate_in_text_citation(
        self, key: str, style: Optional[CitationStyle] = None, page: Optional[str] = None
    ) -> str:
        """
        Generate an in-text citation.

        Args:
            key: Citation key
            style: Citation style (uses default if not specified)
            page: Optional page number for direct quotes

        Returns:
            In-text citation string
        """
        citation = self.citations.get(key)
        if not citation:
            return f"[{key}]"

        style = style or self.default_style

        # Get first author's last name
        if citation.authors:
            first_author = citation.authors[0]
            if "," in first_author:
                last_name = first_author.split(",")[0].strip()
            else:
                last_name = first_author.split()[-1]
        else:
            last_name = "Unknown"

        year = citation.year or "n.d."

        if style in [CitationStyle.APA, CitationStyle.APA7]:
            if len(citation.authors) == 1:
                base = f"({last_name}, {year})"
            elif len(citation.authors) == 2:
                author2 = citation.authors[1]
                if "," in author2:
                    last2 = author2.split(",")[0].strip()
                else:
                    last2 = author2.split()[-1]
                base = f"({last_name} & {last2}, {year})"
            else:
                base = f"({last_name} et al., {year})"

            if page:
                base = base[:-1] + f", p. {page})"
            return base

        elif style in [CitationStyle.MLA, CitationStyle.MLA9]:
            if page:
                return f"({last_name} {page})"
            return f"({last_name})"

        elif style == CitationStyle.IEEE:
            # IEEE uses numbered citations
            if key in self._citation_order:
                num = self._citation_order.index(key) + 1
                return f"[{num}]"
            return f"[{key}]"

        elif style == CitationStyle.VANCOUVER:
            # Vancouver also uses numbered citations
            if key in self._citation_order:
                num = self._citation_order.index(key) + 1
                return f"({num})"
            return f"({key})"

        else:
            # Default format
            return f"({last_name}, {year})"

    def generate_bibliography(
        self,
        style: Optional[CitationStyle] = None,
        sort_by: str = "author",  # "author", "year", "order"
    ) -> str:
        """
        Generate a formatted bibliography.

        Args:
            style: Citation style
            sort_by: Sort order ("author", "year", "order")

        Returns:
            Formatted bibliography string
        """
        style = style or self.default_style

        # Sort citations
        if sort_by == "author":
            sorted_keys = sorted(
                self.citations.keys(),
                key=lambda k: (
                    self.citations[k].authors[0] if self.citations[k].authors else "ZZZ"
                ).lower(),
            )
        elif sort_by == "year":
            sorted_keys = sorted(
                self.citations.keys(), key=lambda k: self.citations[k].year or 9999
            )
        else:  # order
            sorted_keys = self._citation_order

        # Format each citation
        entries = []
        for i, key in enumerate(sorted_keys, 1):
            formatted = self.format_citation(key, style)
            if formatted:
                if style in [CitationStyle.IEEE, CitationStyle.VANCOUVER]:
                    entries.append(f"[{i}] {formatted}")
                else:
                    entries.append(formatted)

        return "\n\n".join(entries)

    def export_bibtex(self) -> str:
        """Export all citations as BibTeX"""
        entries = []
        for key in self._citation_order:
            citation = self.citations[key]
            entries.append(CitationFormatter.format(citation, CitationStyle.BIBTEX))

        return "\n\n".join(entries)

    def to_dict(self) -> Dict[str, Any]:
        """Export citation manager state as dictionary"""
        return {
            "default_style": self.default_style.value,
            "citations": {k: v.model_dump() for k, v in self.citations.items()},
            "order": self._citation_order,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CitationManager":
        """Create citation manager from dictionary"""
        manager = cls(default_style=CitationStyle(data.get("default_style", "apa7")))

        for key, citation_data in data.get("citations", {}).items():
            citation = Citation(**citation_data)
            citation.citation_key = key
            manager.citations[key] = citation

        manager._citation_order = data.get("order", list(manager.citations.keys()))

        return manager
