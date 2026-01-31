#!/usr/bin/env python3
"""
Bulk Ingestion Script for RAG Vector Database

Pre-populates the vector database with data from various sources:
- Economic indicators from government APIs
- Academic papers from OpenAlex/CrossRef
- Current events and market data

Usage:
    uv run python scripts/bulk_ingest.py [--economic] [--papers] [--events] [--all]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.ai_research_generator.rag import RAGService, RAGConfig
from src.ai_research_generator.tools.economic_data import EconomicDataTool, EconomicDataConfig
from src.ai_research_generator.tools.current_events import CurrentEventsTool, CurrentEventsConfig
from src.ai_research_generator.tools.research_tools import ToolConfig
from src.ai_research_generator.legacy.academic_search import UnifiedAcademicSearch


async def ingest_economic_data(rag: RAGService) -> int:
    """Ingest economic data from government APIs."""
    logger.info("=== Ingesting Economic Data ===")

    tool_config = ToolConfig()
    economic_tool = EconomicDataTool(tool_config, EconomicDataConfig())

    try:
        report = await economic_tool.execute(
            include_gdp=True,
            include_inflation=True,
            include_employment=True,
            include_interest_rates=True,
            include_markets=True,
            include_debt=True,
            limit=50,
        )

        count = 0
        for ind in report.indicators:
            text = (
                f"Economic Indicator: {ind.name}\n"
                f"Value: {ind.current_value}\n"
                f"Trend: {ind.trend}\n"
                f"Change: {ind.change_percent}%\n"
                f"Source: {ind.source}\n"
                f"Date: {ind.date}"
            )
            rag.ingest_text(
                text,
                metadata={
                    "type": "economic_indicator",
                    "name": ind.name,
                    "source": ind.source,
                    "trend": ind.trend,
                },
            )
            count += 1

        logger.info(f"Ingested {count} economic indicators")
        return count

    except Exception as e:
        logger.error(f"Economic data ingestion failed: {e}")
        return 0


async def ingest_academic_papers(rag: RAGService, topics: list[str]) -> int:
    """Ingest academic papers for given topics."""
    logger.info("=== Ingesting Academic Papers ===")

    academic_search = UnifiedAcademicSearch()

    total_count = 0
    for topic in topics:
        logger.info(f"Searching papers for: {topic}")
        try:
            papers = academic_search.search_merged(
                query=topic,
                limit=20,
                year_range=(2020, 2026),
                sources=["openalex", "crossref"],
            )

            for paper in papers:
                authors_str = ", ".join(paper.authors[:3]) if paper.authors else "Unknown"
                text = (
                    f"Academic Paper: {paper.title}\n"
                    f"Authors: {authors_str}\n"
                    f"Year: {paper.year or 'N/A'}\n"
                    f"Abstract: {paper.abstract[:800] if paper.abstract else 'No abstract'}\n"
                    f"DOI: {paper.doi or 'N/A'}\n"
                    f"Citations: {paper.citation_count or 0}"
                )
                rag.ingest_text(
                    text,
                    metadata={
                        "type": "academic_paper",
                        "title": paper.title,
                        "year": paper.year,
                        "doi": paper.doi,
                        "source": paper.source,
                        "topic": topic,
                    },
                )
                total_count += 1

            logger.info(f"Ingested {len(papers)} papers for '{topic}'")

        except Exception as e:
            logger.warning(f"Paper search failed for '{topic}': {e}")

    academic_search.close()
    logger.info(f"Total papers ingested: {total_count}")
    return total_count


async def ingest_current_events(rag: RAGService, topics: list[str]) -> int:
    """Ingest current events for given topics."""
    logger.info("=== Ingesting Current Events ===")

    tool_config = ToolConfig()
    events_tool = CurrentEventsTool(tool_config, CurrentEventsConfig())

    total_count = 0
    for topic in topics:
        logger.info(f"Fetching events for: {topic}")
        try:
            analysis = await events_tool.execute(
                topic=topic,
                time_range_days=30,
            )

            # Ingest events
            for event in analysis.major_events[:10]:
                text = (
                    f"Current Event: {event.title}\n"
                    f"Date: {event.date}\n"
                    f"Summary: {event.summary}\n"
                    f"Relevance: {event.relevance}"
                )
                rag.ingest_text(
                    text,
                    metadata={
                        "type": "current_event",
                        "title": event.title,
                        "topic": topic,
                    },
                )
                total_count += 1

            # Ingest market conditions
            for mc in analysis.market_conditions:
                text = (
                    f"Market Condition: {mc.indicator}\n"
                    f"Status: {mc.current_state}\n"
                    f"Trend: {mc.trend}\n"
                    f"Impact: {mc.implications}"
                )
                rag.ingest_text(
                    text,
                    metadata={
                        "type": "market_condition",
                        "indicator": mc.indicator,
                        "topic": topic,
                    },
                )
                total_count += 1

            logger.info(f"Ingested events for '{topic}'")

        except Exception as e:
            logger.warning(f"Events fetch failed for '{topic}': {e}")

    logger.info(f"Total events/conditions ingested: {total_count}")
    return total_count


async def main():
    parser = argparse.ArgumentParser(description="Bulk ingest data into RAG vector database")
    parser.add_argument("--economic", action="store_true", help="Ingest economic data")
    parser.add_argument("--papers", action="store_true", help="Ingest academic papers")
    parser.add_argument("--events", action="store_true", help="Ingest current events")
    parser.add_argument("--all", action="store_true", help="Ingest all data sources")
    parser.add_argument(
        "--topics",
        nargs="+",
        default=[
            "investment strategies",
            "portfolio management",
            "economic indicators",
            "inflation trends",
            "interest rate policy",
            "stock market analysis",
            "bond market",
            "retirement planning",
            "asset allocation",
            "risk management",
        ],
        help="Topics to search for papers and events",
    )

    args = parser.parse_args()

    # Default to all if nothing specified
    if not (args.economic or args.papers or args.events or args.all):
        args.all = True

    # Initialize RAG service
    logger.info("Initializing RAG service...")
    rag_config = RAGConfig()
    rag = RAGService(rag_config)

    if not rag.is_available:
        logger.error("RAG service not available. Check Ollama and ChromaDB.")
        sys.exit(1)

    # Show initial stats
    stats = rag.get_collection_stats()
    logger.info(f"Initial collection stats: {stats}")

    total_ingested = 0

    # Run ingestion
    if args.all or args.economic:
        count = await ingest_economic_data(rag)
        total_ingested += count

    if args.all or args.papers:
        count = await ingest_academic_papers(rag, args.topics)
        total_ingested += count

    if args.all or args.events:
        count = await ingest_current_events(rag, args.topics[:5])  # Limit events topics
        total_ingested += count

    # Show final stats
    stats = rag.get_collection_stats()
    logger.info("=== Ingestion Complete ===")
    logger.info(f"Total items ingested: {total_ingested}")
    logger.info(f"Final collection stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
