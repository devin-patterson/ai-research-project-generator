#!/usr/bin/env python3
"""
Research Engine Module

Integrates LLM capabilities with academic search to provide comprehensive
AI-powered research project generation with real paper discovery and analysis.

Author: AI Research Assistant
Date: 2025-01-29
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from llm_provider import (
    LLMConfig, 
    LLMProvider, 
    ResearchLLMAssistant,
    create_llm_client,
    get_model_recommendations,
    RECOMMENDED_MODELS
)
from academic_search import (
    UnifiedAcademicSearch,
    Paper,
    SearchResult
)
from ai_research_project_generator import (
    AIResearchProjectGenerator,
    ResearchContext,
    ResearchType,
    AcademicLevel,
    ResearchProject
)
from subject_analyzer import SubjectAnalyzer, SubjectAnalysis
from validation_engine import ValidationEngine, ValidationReport


@dataclass
class EnhancedResearchProject:
    """Enhanced research project with AI-generated content and real papers"""
    base_project: Dict[str, Any]
    subject_analysis: Dict[str, Any]
    validation_report: Dict[str, Any]
    
    # AI-enhanced content
    ai_topic_analysis: Optional[str] = None
    ai_research_questions: Optional[List[str]] = None
    ai_methodology_recommendations: Optional[str] = None
    ai_search_strategy: Optional[str] = None
    ai_literature_synthesis: Optional[str] = None
    
    # Real papers from academic search
    discovered_papers: Optional[List[Dict[str, Any]]] = None
    paper_statistics: Optional[Dict[str, Any]] = None
    
    # Metadata
    llm_model_used: Optional[str] = None
    search_sources_used: Optional[List[str]] = None
    generated_at: str = ""
    enhanced: bool = False


class AIResearchEngine:
    """
    Comprehensive AI-powered research engine that combines:
    - Local LLM for intelligent analysis and generation
    - Academic search APIs for real paper discovery
    - Rule-based project generation for structure
    - Validation for quality assurance
    """
    
    def __init__(self, 
                 llm_config: Optional[LLMConfig] = None,
                 semantic_scholar_key: Optional[str] = None,
                 openalex_email: Optional[str] = None,
                 crossref_email: Optional[str] = None,
                 use_llm: bool = True,
                 use_academic_search: bool = True):
        """
        Initialize the AI Research Engine
        
        Args:
            llm_config: Configuration for LLM provider (default: Ollama)
            semantic_scholar_key: API key for Semantic Scholar
            openalex_email: Email for OpenAlex polite pool
            crossref_email: Email for CrossRef polite pool
            use_llm: Whether to use LLM for enhanced analysis
            use_academic_search: Whether to search academic databases
        """
        self.use_llm = use_llm
        self.use_academic_search = use_academic_search
        
        # Initialize base components
        self.project_generator = AIResearchProjectGenerator()
        self.subject_analyzer = SubjectAnalyzer()
        self.validation_engine = ValidationEngine()
        
        # Initialize LLM if enabled
        self.llm_assistant = None
        if use_llm:
            try:
                self.llm_assistant = ResearchLLMAssistant(llm_config)
                logger.info(f"LLM initialized: {llm_config.model if llm_config else 'default'}")
            except Exception as e:
                logger.warning(f"Could not initialize LLM: {e}. Continuing without LLM.")
                self.use_llm = False
        
        # Initialize academic search if enabled
        self.academic_search = None
        if use_academic_search:
            self.academic_search = UnifiedAcademicSearch(
                semantic_scholar_key=semantic_scholar_key,
                openalex_email=openalex_email,
                crossref_email=crossref_email
            )
            logger.info("Academic search initialized")
    
    def generate_enhanced_project(
        self,
        topic: str,
        research_question: str,
        research_type: str,
        academic_level: str,
        discipline: str,
        search_papers: bool = True,
        paper_limit: int = 20,
        year_range: Optional[tuple] = None,
        **kwargs
    ) -> EnhancedResearchProject:
        """
        Generate a comprehensive AI-enhanced research project
        
        Args:
            topic: Research topic
            research_question: Primary research question
            research_type: Type of research methodology
            academic_level: Academic level
            discipline: Academic discipline
            search_papers: Whether to search for real papers
            paper_limit: Maximum number of papers to retrieve
            year_range: Year range for paper search (default: last 5 years)
            **kwargs: Additional parameters for project generation
        
        Returns:
            EnhancedResearchProject with AI analysis and real papers
        """
        logger.info(f"Generating enhanced research project: {topic}")
        
        # Set default year range
        if year_range is None:
            current_year = datetime.now().year
            year_range = (current_year - 5, current_year)
        
        # Step 1: Generate base project structure
        logger.info("Step 1: Generating base project structure...")
        context = ResearchContext(
            topic=topic,
            research_question=research_question,
            research_type=ResearchType(research_type),
            academic_level=AcademicLevel(academic_level),
            discipline=discipline,
            target_publication=kwargs.get("target_publication"),
            citation_style=kwargs.get("citation_style", "APA"),
            time_constraint=kwargs.get("time_constraint"),
            word_count_target=kwargs.get("word_count_target"),
            geographical_scope=kwargs.get("geographical_scope"),
            language=kwargs.get("language", "English")
        )
        
        base_project = self.project_generator.generate_research_project(context)
        
        # Step 2: Perform subject analysis
        logger.info("Step 2: Performing subject analysis...")
        subject_analysis = self.subject_analyzer.analyze_subject(
            topic, research_question, discipline
        )
        
        # Step 3: Validate project
        logger.info("Step 3: Validating project...")
        project_data = {
            "context": asdict(context),
            "literature_search": asdict(base_project.literature_search),
            "methodology": asdict(base_project.methodology),
            "research_questions": base_project.research_questions,
            "timeline": base_project.timeline,
            "expected_outcomes": base_project.expected_outcomes
        }
        validation_report = self.validation_engine.validate_project(project_data)
        
        # Initialize enhanced project
        enhanced_project = EnhancedResearchProject(
            base_project=asdict(base_project),
            subject_analysis=asdict(subject_analysis),
            validation_report=validation_report.__dict__,
            generated_at=datetime.now().isoformat(),
            enhanced=False
        )
        
        # Step 4: Search for real papers
        discovered_papers = []
        if self.use_academic_search and search_papers:
            logger.info("Step 4: Searching academic databases...")
            try:
                papers = self.academic_search.search_merged(
                    query=topic,
                    limit=paper_limit,
                    year_range=year_range,
                    sources=["semantic_scholar", "openalex", "crossref"]
                )
                
                discovered_papers = [asdict(p) for p in papers]
                enhanced_project.discovered_papers = discovered_papers
                enhanced_project.search_sources_used = ["semantic_scholar", "openalex", "crossref"]
                
                # Calculate paper statistics
                enhanced_project.paper_statistics = self._calculate_paper_statistics(papers)
                
                logger.info(f"Found {len(papers)} papers")
            except Exception as e:
                logger.error(f"Error searching papers: {e}")
        
        # Step 5: Enhance with LLM analysis
        if self.use_llm and self.llm_assistant:
            logger.info("Step 5: Enhancing with LLM analysis...")
            try:
                # Topic analysis
                topic_result = self.llm_assistant.analyze_topic(topic, discipline)
                enhanced_project.ai_topic_analysis = topic_result["analysis"]
                enhanced_project.llm_model_used = topic_result.get("model")
                
                # Generate research questions
                ai_questions = self.llm_assistant.generate_research_questions(
                    topic, research_type, research_question
                )
                enhanced_project.ai_research_questions = ai_questions
                
                # Methodology recommendations
                method_result = self.llm_assistant.recommend_methodology(
                    topic, research_question, discipline
                )
                enhanced_project.ai_methodology_recommendations = method_result["recommendations"]
                
                # Search strategy
                search_result = self.llm_assistant.generate_search_strategy(topic, discipline)
                enhanced_project.ai_search_strategy = search_result["strategy"]
                
                # Literature synthesis (if papers found)
                if discovered_papers:
                    paper_dicts = [
                        {"title": p["title"], "authors": ", ".join(p["authors"][:3]), "abstract": p.get("abstract", "")}
                        for p in discovered_papers[:10]
                    ]
                    synthesis = self.llm_assistant.synthesize_findings(paper_dicts, research_question)
                    enhanced_project.ai_literature_synthesis = synthesis
                
                enhanced_project.enhanced = True
                logger.info("LLM enhancement complete")
                
            except Exception as e:
                logger.error(f"Error during LLM enhancement: {e}")
        
        return enhanced_project
    
    def _calculate_paper_statistics(self, papers: List[Paper]) -> Dict[str, Any]:
        """Calculate statistics about discovered papers"""
        if not papers:
            return {}
        
        years = [p.year for p in papers if p.year]
        citations = [p.citation_count for p in papers if p.citation_count]
        sources = [p.source for p in papers]
        
        stats = {
            "total_papers": len(papers),
            "papers_with_abstracts": sum(1 for p in papers if p.abstract),
            "papers_with_pdf": sum(1 for p in papers if p.pdf_url),
            "year_range": {
                "min": min(years) if years else None,
                "max": max(years) if years else None,
                "median": sorted(years)[len(years)//2] if years else None
            },
            "citation_stats": {
                "total": sum(citations) if citations else 0,
                "average": sum(citations) / len(citations) if citations else 0,
                "max": max(citations) if citations else 0,
                "min": min(citations) if citations else 0
            },
            "sources": {source: sources.count(source) for source in set(sources)}
        }
        
        return stats
    
    def export_enhanced_project(self, project: EnhancedResearchProject, 
                                format: str = "markdown") -> str:
        """Export enhanced project in specified format"""
        if format.lower() == "json":
            return json.dumps(asdict(project), indent=2, default=str)
        elif format.lower() == "markdown":
            return self._export_markdown(project)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_markdown(self, project: EnhancedResearchProject) -> str:
        """Export enhanced project as comprehensive markdown"""
        base = project.base_project
        analysis = project.subject_analysis
        validation = project.validation_report
        
        md = f"""# AI-Enhanced Research Project

## Project Overview

**Topic**: {base['context']['topic']}  
**Research Question**: {base['context']['research_question']}  
**Research Type**: {base['context']['research_type']}  
**Academic Level**: {base['context']['academic_level']}  
**Discipline**: {base['context']['discipline']}  
**Generated**: {project.generated_at}  
**AI Enhanced**: {'Yes' if project.enhanced else 'No'}  
**LLM Model**: {project.llm_model_used or 'N/A'}

---

"""
        
        # AI Topic Analysis
        if project.ai_topic_analysis:
            md += f"""## 🤖 AI Topic Analysis

{project.ai_topic_analysis}

---

"""
        
        # AI Research Questions
        if project.ai_research_questions:
            md += """## 🤖 AI-Generated Research Questions

"""
            for i, q in enumerate(project.ai_research_questions, 1):
                md += f"{i}. {q}\n"
            md += "\n---\n\n"
        
        # AI Methodology Recommendations
        if project.ai_methodology_recommendations:
            md += f"""## 🤖 AI Methodology Recommendations

{project.ai_methodology_recommendations}

---

"""
        
        # AI Search Strategy
        if project.ai_search_strategy:
            md += f"""## 🤖 AI-Generated Search Strategy

{project.ai_search_strategy}

---

"""
        
        # Discovered Papers
        if project.discovered_papers:
            stats = project.paper_statistics or {}
            md += f"""## 📚 Discovered Papers

**Total Papers Found**: {stats.get('total_papers', len(project.discovered_papers))}  
**Papers with Abstracts**: {stats.get('papers_with_abstracts', 'N/A')}  
**Papers with PDF**: {stats.get('papers_with_pdf', 'N/A')}  
**Year Range**: {stats.get('year_range', {}).get('min', 'N/A')} - {stats.get('year_range', {}).get('max', 'N/A')}  
**Total Citations**: {stats.get('citation_stats', {}).get('total', 'N/A')}  
**Sources**: {', '.join(f"{k}: {v}" for k, v in stats.get('sources', {}).items())}

### Top Papers by Citations

"""
            # Sort by citations and show top 10
            sorted_papers = sorted(
                project.discovered_papers, 
                key=lambda p: p.get('citation_count') or 0, 
                reverse=True
            )[:10]
            
            for i, paper in enumerate(sorted_papers, 1):
                authors = ", ".join(paper.get('authors', [])[:3])
                if len(paper.get('authors', [])) > 3:
                    authors += " et al."
                
                md += f"""### {i}. {paper.get('title', 'Unknown Title')}

**Authors**: {authors}  
**Year**: {paper.get('year', 'N/A')}  
**Citations**: {paper.get('citation_count', 'N/A')}  
**Source**: {paper.get('source', 'N/A')}  
"""
                if paper.get('doi'):
                    md += f"**DOI**: https://doi.org/{paper['doi']}  \n"
                if paper.get('pdf_url'):
                    md += f"**PDF**: {paper['pdf_url']}  \n"
                if paper.get('abstract'):
                    abstract = paper['abstract'][:500] + "..." if len(paper.get('abstract', '')) > 500 else paper.get('abstract', '')
                    md += f"\n**Abstract**: {abstract}\n"
                md += "\n"
            
            md += "---\n\n"
        
        # AI Literature Synthesis
        if project.ai_literature_synthesis:
            md += f"""## 🤖 AI Literature Synthesis

{project.ai_literature_synthesis}

---

"""
        
        # Base Project Details
        md += f"""## 📋 Research Project Structure

### Literature Search Strategy

**Databases**: {', '.join(base['literature_search']['databases'])}

**Keywords**: {', '.join(base['literature_search']['keywords'])}

**Inclusion Criteria**:
{chr(10).join(f"- {c}" for c in base['literature_search']['inclusion_criteria'])}

**Exclusion Criteria**:
{chr(10).join(f"- {c}" for c in base['literature_search']['exclusion_criteria'])}

### Methodology

**Approach**: {base['methodology']['approach']}  
**Design**: {base['methodology']['design']}  
**Theoretical Framework**: {base['methodology'].get('theoretical_framework', 'N/A')}

**Data Collection Methods**:
{chr(10).join(f"- {m}" for m in base['methodology']['data_collection_methods'])}

**Analysis Methods**:
{chr(10).join(f"- {m}" for m in base['methodology']['analysis_methods'])}

### Research Questions (Rule-Based)

{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(base['research_questions']))}

### Expected Outcomes

{chr(10).join(f"- {o}" for o in base['expected_outcomes'])}

### Timeline

{chr(10).join(f"**{phase}**: {duration}" for phase, duration in base['timeline'].items())}

---

## ✅ Validation Report

**Quality Score**: {validation.get('overall_score', 0):.2f}/1.00  
**Total Issues**: {len(validation.get('issues', []))}

### Recommendations

{chr(10).join(f"- {r}" for r in validation.get('recommendations', []))}

### Enhancement Opportunities

{chr(10).join(f"- {o}" for o in validation.get('enhancement_opportunities', []))}

---

*Generated by AI Research Engine*
"""
        
        return md
    
    def close(self):
        """Close all connections"""
        if self.llm_assistant:
            self.llm_assistant.close()
        if self.academic_search:
            self.academic_search.close()


def print_model_recommendations():
    """Print recommended models for different use cases"""
    print("\n" + "="*60)
    print("RECOMMENDED LOCAL LLM MODELS FOR RESEARCH")
    print("="*60)
    
    for category, data in RECOMMENDED_MODELS.items():
        print(f"\n### {category.upper().replace('_', ' ')}")
        print(f"    {data['description']}")
        print()
        for model in data["ollama"]:
            print(f"    - {model['name']}")
            print(f"      VRAM: {model['vram']} | Context: {model['context']}")
            print(f"      Notes: {model['notes']}")


def main():
    """Example usage of the AI Research Engine"""
    print_model_recommendations()
    
    print("\n" + "="*60)
    print("GENERATING ENHANCED RESEARCH PROJECT")
    print("="*60)
    
    # Initialize engine (will work without LLM if Ollama not running)
    engine = AIResearchEngine(
        llm_config=LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.1:8b",
            base_url="http://localhost:11434"
        ),
        use_llm=True,
        use_academic_search=True
    )
    
    # Generate enhanced project
    project = engine.generate_enhanced_project(
        topic="Impact of remote work on employee productivity and well-being",
        research_question="How has the shift to remote work affected employee productivity and mental health outcomes?",
        research_type="systematic_review",
        academic_level="graduate",
        discipline="psychology",
        search_papers=True,
        paper_limit=15,
        year_range=(2020, 2025)
    )
    
    # Export and print
    output = engine.export_enhanced_project(project, "markdown")
    print(output)
    
    # Save to file
    with open("enhanced_research_project.md", "w") as f:
        f.write(output)
    print("\nProject saved to: enhanced_research_project.md")
    
    engine.close()


if __name__ == "__main__":
    main()
