#!/usr/bin/env python3
"""
AI Research Project Generator - Main Application

A comprehensive solution for generating robust research projects using AI.
This main module integrates all components to provide detailed and accurate
subject analysis for research projects.

Author: AI Research Assistant
Date: 2025-01-29
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Optional
from dataclasses import asdict

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from .ai_research_project_generator import (
    AIResearchProjectGenerator,
    ResearchContext,
    ResearchType,
    AcademicLevel,
)
from .subject_analyzer import SubjectAnalyzer
from .validation_engine import ValidationEngine


class IntegratedResearchGenerator:
    """Integrated research project generation system"""

    def __init__(self):
        """Initialize the integrated system"""
        self.project_generator = AIResearchProjectGenerator()
        self.subject_analyzer = SubjectAnalyzer()
        self.validation_engine = ValidationEngine()

    def generate_comprehensive_project(
        self,
        topic: str,
        research_question: str,
        research_type: str,
        academic_level: str,
        discipline: str,
        **kwargs,
    ) -> Dict:
        """
        Generate a comprehensive research project with analysis and validation

        Args:
            topic: Research topic
            research_question: Primary research question
            research_type: Type of research
            academic_level: Academic level
            discipline: Academic discipline
            **kwargs: Additional parameters

        Returns:
            Complete research project with analysis and validation
        """
        logger.info(f"Generating comprehensive research project: {topic}")

        # Create research context
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
            language=kwargs.get("language", "English"),
        )

        # Generate research project
        project = self.project_generator.generate_research_project(context)

        # Perform subject analysis
        subject_analysis = self.subject_analyzer.analyze_subject(
            topic, research_question, discipline
        )

        # Validate project
        project_data = {
            "context": asdict(context),
            "literature_search": asdict(project.literature_search),
            "methodology": asdict(project.methodology),
            "research_questions": project.research_questions,
            "timeline": project.timeline,
            "expected_outcomes": project.expected_outcomes,
        }

        validation_report = self.validation_engine.validate_project(project_data)

        # Combine all results
        comprehensive_project = {
            "project": asdict(project),
            "subject_analysis": asdict(subject_analysis),
            "validation_report": validation_report.__dict__,
            "metadata": {
                "generated_at": project.created_at,
                "quality_score": validation_report.overall_score,
                "total_issues": len(validation_report.issues),
                "critical_issues": len(
                    [i for i in validation_report.issues if i.level.value == "critical"]
                ),
            },
        }

        logger.info(
            f"Comprehensive project generated. Quality score: {validation_report.overall_score:.2f}"
        )
        return comprehensive_project

    def export_project(
        self, project: Dict, format: str = "json", output_path: Optional[str] = None
    ) -> str:
        """
        Export comprehensive project in specified format

        Args:
            project: Comprehensive project data
            format: Export format (json, markdown, html)
            output_path: Optional output file path

        Returns:
            Exported project as string
        """
        if format.lower() == "json":
            content = json.dumps(project, indent=2, default=str)
        elif format.lower() == "markdown":
            content = self._export_markdown(project)
        elif format.lower() == "html":
            content = self._export_html(project)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if output_path:
            Path(output_path).write_text(content, encoding="utf-8")
            logger.info(f"Project exported to {output_path}")

        return content

    def _export_markdown(self, project: Dict) -> str:
        """Export project as comprehensive markdown"""
        project_data = project["project"]
        analysis_data = project["subject_analysis"]
        validation_data = project["validation_report"]
        metadata = project["metadata"]

        md = f"""# AI-Generated Research Project

## Project Overview
**Topic**: {project_data["context"]["topic"]}  
**Research Question**: {project_data["context"]["research_question"]}  
**Research Type**: {project_data["context"]["research_type"]}  
**Academic Level**: {project_data["context"]["academic_level"]}  
**Discipline**: {project_data["context"]["discipline"]}  
**Quality Score**: {metadata["quality_score"]:.2f}/1.00  
**Generated**: {metadata["generated_at"]}

---

## Research Context

### Topic Details
{project_data["context"]["topic"]}

### Primary Research Question
{project_data["context"]["research_question"]}

### Research Parameters
- **Type**: {project_data["context"]["research_type"]}
- **Level**: {project_data["context"]["academic_level"]}
- **Discipline**: {project_data["context"]["discipline"]}
- **Target Publication**: {project_data["context"].get("target_publication", "Not specified")}
- **Citation Style**: {project_data["context"]["citation_style"]}

---

## Subject Analysis

### Primary Concepts
"""

        # Add primary concepts
        for concept in analysis_data.get("primary_concepts", []):
            md += f"""
#### {concept["name"]}
- **Definition**: {concept["definition"]}
- **Importance**: {concept["importance"]:.2f}
- **Related Concepts**: {", ".join(concept["related_concepts"])}
- **Measurement Indicators**: {", ".join(concept["measurement_indicators"])}
"""

        md += """
### Methodology Recommendations
"""

        # Add methodology recommendations
        for method in analysis_data.get("methodology_recommendations", []):
            md += f"""
#### {method["name"]}
**Rationale**: {method["rationale"]}

**Strengths**:
{chr(10).join(f"- {s}" for s in method["strengths"])}

**Limitations**:
{chr(10).join(f"- {limitation}" for limitation in method["limitations"])}

**Data Requirements**: {", ".join(method["data_requirements"])}
**Analysis Techniques**: {", ".join(method["analysis_techniques"])}
"""

        md += f"""
### Key Variables
{chr(10).join(f"- {var}" for var in analysis_data.get("key_variables", []))}

### Theoretical Frameworks
{chr(10).join(f"- {fw}" for fw in analysis_data.get("theoretical_frameworks", []))}

### Potential Challenges
{chr(10).join(f"- {challenge}" for challenge in analysis_data.get("potential_challenges", []))}

### Research Gaps
{chr(10).join(f"- {gap}" for gap in analysis_data.get("research_gaps", []))}

---

## Literature Search Strategy

### Databases
{chr(10).join(f"- {db}" for db in project_data["literature_search"]["databases"])}

### Keywords
{chr(10).join(f"- {keyword}" for keyword in project_data["literature_search"]["keywords"])}

### Inclusion Criteria
{chr(10).join(f"- {criteria}" for criteria in project_data["literature_search"]["inclusion_criteria"])}

### Exclusion Criteria
{chr(10).join(f"- {criteria}" for criteria in project_data["literature_search"]["exclusion_criteria"])}

---

## Research Methodology

**Approach**: {project_data["methodology"]["approach"]}  
**Design**: {project_data["methodology"]["design"]}  
**Theoretical Framework**: {project_data["methodology"].get("theoretical_framework", "Not specified")}

### Data Collection Methods
{chr(10).join(f"- {method}" for method in project_data["methodology"]["data_collection_methods"])}

### Analysis Methods
{chr(10).join(f"- {method}" for method in project_data["methodology"]["analysis_methods"])}

### Ethical Considerations
{chr(10).join(f"- {consideration}" for consideration in project_data["methodology"].get("ethical_considerations", []))}

---

## Research Questions
{chr(10).join(f"{i + 1}. {q}" for i, q in enumerate(project_data["research_questions"]))}

---

## Expected Outcomes
{chr(10).join(f"- {outcome}" for outcome in project_data["expected_outcomes"])}

---

## Research Timeline
{chr(10).join(f"**{phase}**: {duration}" for phase, duration in project_data["timeline"].items())}

---

## Validation Report

### Quality Assessment
**Overall Score**: {validation_data["overall_score"]:.2f}/1.00  
**Total Issues**: {len(validation_data["issues"])}  
**Critical Issues**: {len([i for i in validation_data["issues"] if i["level"] == "critical"])}  
**Warnings**: {len([i for i in validation_data["issues"] if i["level"] == "warning"])}

### Key Issues
"""

        # Add critical issues
        critical_issues = [i for i in validation_data["issues"] if i["level"] == "critical"]
        if critical_issues:
            md += "\n#### 🚨 Critical Issues\n"
            for issue in critical_issues[:5]:  # Limit to top 5
                md += f"- **{issue['title']}**: {issue['description']}\n"

        # Add warnings
        warning_issues = [i for i in validation_data["issues"] if i["level"] == "warning"]
        if warning_issues:
            md += "\n#### ⚠️ Warnings\n"
            for issue in warning_issues[:5]:  # Limit to top 5
                md += f"- **{issue['title']}**: {issue['description']}\n"

        md += f"""
### Recommendations
{chr(10).join(f"- {rec}" for rec in validation_data["recommendations"])}

### Enhancement Opportunities
{chr(10).join(f"- {opp}" for opp in validation_data["enhancement_opportunities"])}

---

## References
{chr(10).join(f"- {ref['authors']} ({ref['year']}). {ref['title']}. *{ref['journal']}*. DOI: {ref['doi']}" for ref in project_data["references"])}

---

*This research project was generated using AI Research Project Generator on {metadata["generated_at"]}.*
"""

        return md

    def _export_html(self, project: Dict) -> str:
        """Export project as HTML"""
        # Convert markdown to HTML (simple implementation)
        markdown_content = self._export_markdown(project)

        # Basic markdown to HTML conversion
        html = markdown_content.replace("# ", "<h1>").replace("\n# ", "\n<h1>")
        html = html.replace("## ", "<h2>").replace("\n## ", "\n<h2>")
        html = html.replace("### ", "<h3>").replace("\n### ", "\n<h3>")
        html = html.replace("#### ", "<h4>").replace("\n#### ", "\n<h4>")
        html = html.replace("- ", "<li>").replace("\n- ", "\n<li>")
        html = html.replace("**", "<strong>").replace("**", "</strong>")
        html = html.replace("*", "<em>").replace("*", "</em>")

        # Wrap in HTML structure
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Research Project: {project["project"]["context"]["topic"]}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
        h2 {{ color: #34495e; }}
        h3 {{ color: #7f8c8d; }}
        .critical {{ color: #e74c3c; }}
        .warning {{ color: #f39c12; }}
        .score {{ font-size: 1.2em; font-weight: bold; color: #27ae60; }}
        .metadata {{ background: #ecf0f1; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
{html}
</body>
</html>"""

        return full_html


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="AI Research Project Generator")
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument("--question", required=True, help="Primary research question")
    parser.add_argument(
        "--type", required=True, choices=[t.value for t in ResearchType], help="Research type"
    )
    parser.add_argument(
        "--level",
        required=True,
        choices=[level.value for level in AcademicLevel],
        help="Academic level",
    )
    parser.add_argument("--discipline", required=True, help="Academic discipline")
    parser.add_argument("--publication", help="Target publication venue")
    parser.add_argument("--citation", default="APA", help="Citation style")
    parser.add_argument(
        "--format", default="markdown", choices=["json", "markdown", "html"], help="Output format"
    )
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="INFO")

    # Initialize the integrated system
    generator = IntegratedResearchGenerator()

    # Generate comprehensive project
    project = generator.generate_comprehensive_project(
        topic=args.topic,
        research_question=args.question,
        research_type=args.type,
        academic_level=args.level,
        discipline=args.discipline,
        target_publication=args.publication,
        citation_style=args.citation,
    )

    # Export project
    content = generator.export_project(project, args.format, args.output)

    if not args.output:
        print(content)

    # Print summary
    metadata = project["metadata"]
    print("\n=== PROJECT SUMMARY ===")
    print(f"Quality Score: {metadata['quality_score']:.2f}/1.00")
    print(f"Total Issues: {metadata['total_issues']}")
    print(f"Critical Issues: {metadata['critical_issues']}")

    if args.output:
        print(f"Project exported to: {args.output}")


if __name__ == "__main__":
    main()
