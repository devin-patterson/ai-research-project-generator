#!/usr/bin/env python3
"""
AI Research Project Generator

A comprehensive solution for generating robust research projects using AI.
This tool implements best practices for research methodology, systematic reviews,
and academic writing to provide detailed and accurate subject analysis.

Author: AI Research Assistant
Date: 2025-01-29
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re

try:
    from loguru import logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class ResearchType(Enum):
    """Types of research methodologies supported"""

    SYSTEMATIC_REVIEW = "systematic_review"
    SCOPING_REVIEW = "scoping_review"
    META_ANALYSIS = "meta_analysis"
    QUALITATIVE_STUDY = "qualitative_study"
    QUANTITATIVE_STUDY = "quantitative_study"
    MIXED_METHODS = "mixed_methods"
    CASE_STUDY = "case_study"
    EXPERIMENTAL = "experimental"


class AcademicLevel(Enum):
    """Academic levels for research projects"""

    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    DOCTORAL = "doctoral"
    POST_DOCTORAL = "post_doctoral"
    PROFESSIONAL = "professional"


@dataclass
class ResearchContext:
    """Research context and parameters"""

    topic: str
    research_question: str
    research_type: ResearchType
    academic_level: AcademicLevel
    discipline: str
    target_publication: Optional[str] = None
    citation_style: str = "APA"
    time_constraint: Optional[str] = None
    word_count_target: Optional[int] = None
    geographical_scope: Optional[str] = None
    language: str = "English"


@dataclass
class LiteratureSearchStrategy:
    """Literature search strategy configuration"""

    databases: List[str]
    keywords: List[str]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    date_range: Optional[Tuple[str, str]] = None
    study_types: List[str] = None
    sample_size_min: Optional[int] = None


@dataclass
class ResearchMethodology:
    """Research methodology configuration"""

    approach: str
    design: str
    data_collection_methods: List[str]
    analysis_methods: List[str]
    theoretical_framework: Optional[str] = None
    ethical_considerations: List[str] = None
    limitations: List[str] = None


@dataclass
class ResearchProject:
    """Complete research project structure"""

    context: ResearchContext
    literature_search: LiteratureSearchStrategy
    methodology: ResearchMethodology
    research_questions: List[str]
    hypotheses: List[str]
    expected_outcomes: List[str]
    timeline: Dict[str, str]
    references: List[Dict[str, str]]
    created_at: str
    updated_at: str


class AIResearchProjectGenerator:
    """Main class for generating AI-powered research projects"""

    def __init__(self):
        """Initialize the research project generator"""
        self.prisma_checklist = self._load_prisma_checklist()
        self.methodology_templates = self._load_methodology_templates()
        self.discipline_databases = self._load_discipline_databases()

    def _load_prisma_checklist(self) -> List[str]:
        """Load PRISMA 2020 checklist items"""
        return [
            "TITLE: Identify the report as a systematic review",
            "ABSTRACT: Provide structured summary",
            "INTRODUCTION: Rationale for review",
            "INTRODUCTION: Objectives with PICO framework",
            "METHODS: Eligibility criteria",
            "METHODS: Information sources",
            "METHODS: Search strategy",
            "METHODS: Selection process",
            "METHODS: Data collection process",
            "METHODS: Data items",
            "METHODS: Risk of bias assessment",
            "METHODS: Effect measures",
            "METHODS: Synthesis methods",
            "METHODS: Reporting bias assessment",
            "METHODS: Certainty assessment",
            "RESULTS: Study selection",
            "RESULTS: Study characteristics",
            "RESULTS: Risk of bias in studies",
            "RESULTS: Results of syntheses",
            "RESULTS: Reporting bias",
            "RESULTS: Certainty of evidence",
            "DISCUSSION: Summary of evidence",
            "DISCUSSION: Limitations",
            "DISCUSSION: Conclusions",
            "FUNDING: Primary source of funding",
            "OTHER: Registration and protocol",
        ]

    def _load_methodology_templates(self) -> Dict[ResearchType, Dict]:
        """Load methodology templates for different research types"""
        return {
            ResearchType.SYSTEMATIC_REVIEW: {
                "approach": "Systematic literature review",
                "design": "PRISMA-guided systematic review",
                "data_collection_methods": ["Database search", "Screening", "Data extraction"],
                "analysis_methods": ["Thematic analysis", "Quality assessment", "Synthesis"],
            },
            ResearchType.QUALITATIVE_STUDY: {
                "approach": "Qualitative inquiry",
                "design": "Phenomenological study",
                "data_collection_methods": ["Semi-structured interviews", "Focus groups"],
                "analysis_methods": [
                    "Thematic analysis",
                    "Interpretative phenomenological analysis",
                ],
            },
            ResearchType.QUANTITATIVE_STUDY: {
                "approach": "Quantitative analysis",
                "design": "Cross-sectional survey",
                "data_collection_methods": ["Surveys", "Questionnaires"],
                "analysis_methods": ["Statistical analysis", "Regression analysis"],
            },
        }

    def _load_discipline_databases(self) -> Dict[str, List[str]]:
        """Load recommended databases by academic discipline"""
        return {
            "medicine": ["PubMed", "MEDLINE", "Cochrane Library", "EMBASE"],
            "psychology": ["PsycINFO", "PubMed", "Google Scholar"],
            "education": ["ERIC", "Google Scholar", "JSTOR"],
            "business": ["Business Source Premier", "ABI/INFORM", "Google Scholar"],
            "computer_science": ["IEEE Xplore", "ACM Digital Library", "arXiv"],
            "social_sciences": ["JSTOR", "Web of Science", "Google Scholar"],
            "engineering": ["IEEE Xplore", "Engineering Village", "Google Scholar"],
        }

    def generate_research_project(self, context: ResearchContext) -> ResearchProject:
        """
        Generate a comprehensive research project based on the given context

        Args:
            context: Research context and parameters

        Returns:
            Complete research project structure
        """
        logger.info(f"Generating research project for: {context.topic}")

        # Generate literature search strategy
        literature_search = self._generate_literature_search(context)

        # Generate methodology
        methodology = self._generate_methodology(context)

        # Generate research questions and hypotheses
        research_questions = self._generate_research_questions(context)
        hypotheses = self._generate_hypotheses(context, research_questions)

        # Generate expected outcomes
        expected_outcomes = self._generate_expected_outcomes(context)

        # Generate timeline
        timeline = self._generate_timeline(context)

        # Generate initial references
        references = self._generate_references(context)

        project = ResearchProject(
            context=context,
            literature_search=literature_search,
            methodology=methodology,
            research_questions=research_questions,
            hypotheses=hypotheses,
            expected_outcomes=expected_outcomes,
            timeline=timeline,
            references=references,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        logger.info("Research project generated successfully")
        return project

    def _generate_literature_search(self, context: ResearchContext) -> LiteratureSearchStrategy:
        """Generate literature search strategy"""
        # Get discipline-specific databases
        discipline_key = context.discipline.lower().replace(" ", "_")
        databases = self.discipline_databases.get(
            discipline_key, ["Google Scholar", "PubMed", "Web of Science"]
        )

        # Generate keywords from topic and research question
        keywords = self._extract_keywords(context.topic, context.research_question)

        # Generate inclusion/exclusion criteria
        inclusion_criteria = self._generate_inclusion_criteria(context)
        exclusion_criteria = self._generate_exclusion_criteria(context)

        return LiteratureSearchStrategy(
            databases=databases,
            keywords=keywords,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            date_range=self._suggest_date_range(context),
            study_types=self._suggest_study_types(context),
            sample_size_min=self._suggest_sample_size(context),
        )

    def _generate_methodology(self, context: ResearchContext) -> ResearchMethodology:
        """Generate research methodology"""
        template = self.methodology_templates.get(
            context.research_type, self.methodology_templates[ResearchType.SYSTEMATIC_REVIEW]
        )

        methodology = ResearchMethodology(
            approach=template["approach"],
            design=template["design"],
            data_collection_methods=template["data_collection_methods"],
            analysis_methods=template["analysis_methods"],
            theoretical_framework=self._suggest_theoretical_framework(context),
            ethical_considerations=self._generate_ethical_considerations(context),
            limitations=self._identify_limitations(context),
        )

        return methodology

    def _extract_keywords(self, topic: str, research_question: str) -> List[str]:
        """Extract keywords from topic and research question"""
        # Combine topic and research question
        text = f"{topic} {research_question}".lower()

        # Extract important terms (simple keyword extraction)
        # In a real implementation, this would use NLP techniques
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
        }

        words = re.findall(r"\b[a-z]+\b", text)
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]

        # Remove duplicates and limit to top keywords
        return list(set(keywords))[:15]

    def _generate_inclusion_criteria(self, context: ResearchContext) -> List[str]:
        """Generate inclusion criteria for literature search"""
        criteria = [
            "Peer-reviewed articles",
            f"Published in {context.language}",
            "Relevant to research topic",
        ]

        if context.academic_level in [AcademicLevel.DOCTORAL, AcademicLevel.POST_DOCTORAL]:
            criteria.append("High-quality journals (Q1/Q2)")

        if context.geographical_scope:
            criteria.append(f"Studies conducted in {context.geographical_scope}")

        return criteria

    def _generate_exclusion_criteria(self, context: ResearchContext) -> List[str]:
        """Generate exclusion criteria for literature search"""
        return [
            "Non-peer reviewed sources",
            "Conference abstracts only",
            "Non-English publications (if English is required)",
            "Studies with methodological flaws",
        ]

    def _suggest_date_range(self, context: ResearchContext) -> Optional[Tuple[str, str]]:
        """Suggest appropriate date range for literature search"""
        current_year = datetime.now().year

        if context.academic_level == AcademicLevel.UNDERGRADUATE:
            return (str(current_year - 10), str(current_year))
        elif context.academic_level == AcademicLevel.GRADUATE:
            return (str(current_year - 15), str(current_year))
        else:  # Doctoral, Post-doctoral, Professional
            return (str(current_year - 20), str(current_year))

    def _suggest_study_types(self, context: ResearchContext) -> List[str]:
        """Suggest appropriate study types"""
        if context.research_type == ResearchType.SYSTEMATIC_REVIEW:
            return [
                "Systematic reviews",
                "Meta-analyses",
                "Randomized controlled trials",
                "Observational studies",
            ]
        elif context.research_type == ResearchType.QUALITATIVE_STUDY:
            return ["Qualitative studies", "Mixed methods studies", "Case studies"]
        elif context.research_type == ResearchType.QUANTITATIVE_STUDY:
            return ["Quantitative studies", "Experimental studies", "Survey research"]
        else:
            return ["Empirical studies", "Review articles"]

    def _suggest_sample_size(self, context: ResearchContext) -> Optional[int]:
        """Suggest minimum sample size for studies"""
        if context.academic_level == AcademicLevel.UNDERGRADUATE:
            return 30
        elif context.academic_level == AcademicLevel.GRADUATE:
            return 50
        else:
            return 100

    def _suggest_theoretical_framework(self, context: ResearchContext) -> Optional[str]:
        """Suggest theoretical framework based on discipline and topic"""
        # This would be enhanced with domain-specific knowledge
        frameworks = {
            "psychology": "Cognitive Behavioral Theory",
            "education": "Constructivist Learning Theory",
            "business": "Resource-Based View",
            "medicine": "Biopsychosocial Model",
            "sociology": "Social Constructivism",
        }

        discipline_key = context.discipline.lower()
        return frameworks.get(discipline_key, "General Systems Theory")

    def _generate_ethical_considerations(self, context: ResearchContext) -> List[str]:
        """Generate ethical considerations for the research"""
        considerations = [
            "Informed consent from participants",
            "Confidentiality and data protection",
            "Ethical approval from institutional review board",
        ]

        if context.research_type in [ResearchType.QUALITATIVE_STUDY, ResearchType.MIXED_METHODS]:
            considerations.append("Anonymization of participant data")

        return considerations

    def _identify_limitations(self, context: ResearchContext) -> List[str]:
        """Identify potential limitations of the research"""
        limitations = [
            "Potential publication bias in literature",
            "Limited by search terms and databases used",
        ]

        if context.time_constraint:
            limitations.append("Time constraints may limit comprehensiveness")

        return limitations

    def _generate_research_questions(self, context: ResearchContext) -> List[str]:
        """Generate focused research questions"""
        main_question = context.research_question

        # Generate sub-questions based on research type
        if context.research_type == ResearchType.SYSTEMATIC_REVIEW:
            sub_questions = [
                f"What are the key themes in {context.topic}?",
                f"What methodologies have been used to study {context.topic}?",
                f"What are the gaps in current knowledge about {context.topic}?",
            ]
        elif context.research_type == ResearchType.QUALITATIVE_STUDY:
            sub_questions = [
                f"What are the lived experiences related to {context.topic}?",
                f"How do individuals perceive {context.topic}?",
                f"What factors influence experiences with {context.topic}?",
            ]
        else:
            sub_questions = [
                f"What is the relationship between key variables in {context.topic}?",
                f"What factors contribute to outcomes in {context.topic}?",
                f"How effective are current interventions for {context.topic}?",
            ]

        return [main_question] + sub_questions

    def _generate_hypotheses(self, context: ResearchContext, questions: List[str]) -> List[str]:
        """Generate testable hypotheses"""
        if context.research_type in [ResearchType.QUALITATIVE_STUDY, ResearchType.SCOPING_REVIEW]:
            return []  # These typically don't have hypotheses

        # Generate hypotheses based on the main research question
        hypotheses = [
            f"There is a significant relationship between key factors in {context.topic}",
            f"Current approaches to {context.topic} have measurable effects",
        ]

        return hypotheses

    def _generate_expected_outcomes(self, context: ResearchContext) -> List[str]:
        """Generate expected research outcomes"""
        outcomes = [
            f"Comprehensive understanding of current knowledge on {context.topic}",
            "Identification of research gaps and future directions",
            "Synthesis of evidence-based recommendations",
        ]

        if context.target_publication:
            outcomes.append(f"Publication suitable for {context.target_publication}")

        return outcomes

    def _generate_timeline(self, context: ResearchContext) -> Dict[str, str]:
        """Generate research timeline"""
        # Default timeline in weeks
        timeline = {
            "Literature Review": "4-6 weeks",
            "Methodology Development": "2-3 weeks",
            "Data Collection": "8-12 weeks",
            "Data Analysis": "4-6 weeks",
            "Writing": "6-8 weeks",
            "Revision": "2-4 weeks",
        }

        # Adjust based on academic level
        if context.academic_level == AcademicLevel.UNDERGRADUATE:
            for key in timeline:
                timeline[key] = timeline[key].replace(
                    "weeks", "weeks"
                )  # Keep same but could be shorter

        return timeline

    def _generate_references(self, context: ResearchContext) -> List[Dict[str, str]]:
        """Generate initial reference list"""
        # This would typically integrate with academic databases
        # For now, return placeholder references
        return [
            {
                "type": "journal_article",
                "title": f"Recent advances in {context.topic}",
                "authors": "Author, A., & Author, B.",
                "journal": context.discipline.capitalize() + " Journal",
                "year": "2023",
                "doi": "10.xxxx/xxxxx",
            },
            {
                "type": "review",
                "title": f"Systematic review of {context.topic}",
                "authors": "Reviewer, C.",
                "journal": "Annual Review of " + context.discipline.capitalize(),
                "year": "2022",
                "doi": "10.xxxx/xxxxx",
            },
        ]

    def export_project(self, project: ResearchProject, format: str = "json") -> str:
        """Export research project in specified format"""
        if format.lower() == "json":
            return json.dumps(asdict(project), indent=2)
        elif format.lower() == "markdown":
            return self._export_markdown(project)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_markdown(self, project: ResearchProject) -> str:
        """Export project as markdown"""
        md = f"""# Research Project: {project.context.topic}

## Research Context
- **Topic**: {project.context.topic}
- **Research Question**: {project.context.research_question}
- **Research Type**: {project.context.research_type.value}
- **Academic Level**: {project.context.academic_level.value}
- **Discipline**: {project.context.discipline}
- **Target Publication**: {project.context.target_publication or "Not specified"}
- **Citation Style**: {project.context.citation_style}

## Literature Search Strategy

### Databases
{chr(10).join(f"- {db}" for db in project.literature_search.databases)}

### Keywords
{chr(10).join(f"- {keyword}" for keyword in project.literature_search.keywords)}

### Inclusion Criteria
{chr(10).join(f"- {criteria}" for criteria in project.literature_search.inclusion_criteria)}

### Exclusion Criteria
{chr(10).join(f"- {criteria}" for criteria in project.literature_search.exclusion_criteria)}

## Research Methodology

- **Approach**: {project.methodology.approach}
- **Design**: {project.methodology.design}
- **Data Collection Methods**: {", ".join(project.methodology.data_collection_methods)}
- **Analysis Methods**: {", ".join(project.methodology.analysis_methods)}
- **Theoretical Framework**: {project.methodology.theoretical_framework}

## Research Questions
{chr(10).join(f"{i + 1}. {q}" for i, q in enumerate(project.research_questions))}

## Hypotheses
{chr(10).join(f"- {h}" for h in project.hypotheses) if project.hypotheses else "Not applicable for this research type"}

## Expected Outcomes
{chr(10).join(f"- {outcome}" for outcome in project.expected_outcomes)}

## Timeline
{chr(10).join(f"**{phase}**: {duration}" for phase, duration in project.timeline.items())}

## References
{chr(10).join(f"- {ref['authors']} ({ref['year']}). {ref['title']}. *{ref['journal']}*. DOI: {ref['doi']}" for ref in project.references)}

---
*Generated on {project.created_at}*
"""
        return md

    def validate_project(self, project: ResearchProject) -> Dict[str, List[str]]:
        """Validate research project for completeness and quality"""
        issues = []
        suggestions = []

        # Check essential components
        if not project.context.research_question:
            issues.append("Research question is missing")

        if not project.literature_search.keywords:
            issues.append("No keywords defined for literature search")

        if len(project.literature_search.databases) < 2:
            suggestions.append("Consider adding more databases for comprehensive search")

        if not project.methodology.data_collection_methods:
            issues.append("No data collection methods specified")

        # Check PRISMA compliance for systematic reviews
        if project.context.research_type == ResearchType.SYSTEMATIC_REVIEW:
            missing_prisma = []
            for item in self.prisma_checklist[:10]:  # Check first 10 critical items
                if item.lower() not in self._export_markdown(project).lower():
                    missing_prisma.append(item)

            if missing_prisma:
                suggestions.append(
                    f"Consider addressing these PRISMA items: {', '.join(missing_prisma[:3])}"
                )

        return {"issues": issues, "suggestions": suggestions}


def main():
    """Example usage of the AI Research Project Generator"""
    generator = AIResearchProjectGenerator()

    # Example research context
    context = ResearchContext(
        topic="Impact of remote work on employee productivity",
        research_question="How has the shift to remote work affected employee productivity and well-being?",
        research_type=ResearchType.SYSTEMATIC_REVIEW,
        academic_level=AcademicLevel.GRADUATE,
        discipline="psychology",
        target_publication="Journal of Occupational Health Psychology",
        citation_style="APA",
        time_constraint="6 months",
        word_count_target=8000,
    )

    # Generate research project
    project = generator.generate_research_project(context)

    # Validate project
    validation = generator.validate_project(project)

    # Export results
    print("=== RESEARCH PROJECT ===")
    print(generator.export_project(project, "markdown"))

    print("\n=== VALIDATION RESULTS ===")
    print("Issues:", validation["issues"])
    print("Suggestions:", validation["suggestions"])


if __name__ == "__main__":
    main()
