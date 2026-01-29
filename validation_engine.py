#!/usr/bin/env python3
"""
Validation Engine

Comprehensive validation and accuracy enhancement features for research projects.
This module provides quality checks, accuracy validation, and enhancement recommendations
for AI-generated research projects.

Author: AI Research Assistant
Date: 2025-01-29
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    SUGGESTION = "suggestion"

class ValidationCategory(Enum):
    """Validation categories"""
    METHODOLOGY = "methodology"
    LITERATURE_SEARCH = "literature_search"
    RESEARCH_QUESTIONS = "research_questions"
    ETHICS = "ethics"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    FEASIBILITY = "feasibility"

@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    category: ValidationCategory
    level: ValidationLevel
    title: str
    description: str
    suggestion: str
    location: Optional[str] = None
    auto_fixable: bool = False

@dataclass
class ValidationReport:
    """Complete validation report"""
    issues: List[ValidationIssue] = field(default_factory=list)
    overall_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    enhancement_opportunities: List[str] = field(default_factory=list)
    validated_at: str = field(default_factory=lambda: datetime.now().isoformat())

class ValidationEngine:
    """Comprehensive validation engine for research projects"""
    
    def __init__(self):
        """Initialize the validation engine"""
        self.methodology_rules = self._load_methodology_rules()
        self.literature_search_rules = self._load_literature_search_rules()
        self.research_question_rules = self._load_research_question_rules()
        self.ethical_guidelines = self._load_ethical_guidelines()
        self.quality_metrics = self._load_quality_metrics()
        
    def _load_methodology_rules(self) -> Dict[str, List[Dict]]:
        """Load methodology validation rules"""
        return {
            "systematic_review": [
                {
                    "rule": "prisma_compliance",
                    "description": "Must follow PRISMA guidelines",
                    "check": self._check_prisma_compliance,
                    "level": ValidationLevel.CRITICAL
                },
                {
                    "rule": "search_strategy",
                    "description": "Must have comprehensive search strategy",
                    "check": self._check_search_strategy_completeness,
                    "level": ValidationLevel.CRITICAL
                },
                {
                    "rule": "quality_assessment",
                    "description": "Must include quality assessment",
                    "check": self._check_quality_assessment,
                    "level": ValidationLevel.WARNING
                }
            ],
            "qualitative_study": [
                {
                    "rule": "theoretical_framework",
                    "description": "Must have clear theoretical framework",
                    "check": self._check_theoretical_framework,
                    "level": ValidationLevel.WARNING
                },
                {
                    "rule": "data_collection_method",
                    "description": "Must specify data collection methods",
                    "check": self._check_data_collection_methods,
                    "level": ValidationLevel.CRITICAL
                },
                {
                    "rule": "trustworthiness",
                    "description": "Must address trustworthiness criteria",
                    "check": self._check_trustworthiness_criteria,
                    "level": ValidationLevel.WARNING
                }
            ],
            "quantitative_study": [
                {
                    "rule": "sample_size",
                    "description": "Must justify sample size",
                    "check": self._check_sample_size_justification,
                    "level": ValidationLevel.WARNING
                },
                {
                    "rule": "statistical_power",
                    "description": "Must consider statistical power",
                    "check": self._check_statistical_power,
                    "level": ValidationLevel.INFO
                },
                {
                    "rule": "measurement_validity",
                    "description": "Must address measurement validity",
                    "check": self._check_measurement_validity,
                    "level": ValidationLevel.WARNING
                }
            ]
        }
    
    def _load_literature_search_rules(self) -> List[Dict]:
        """Load literature search validation rules"""
        return [
            {
                "rule": "database_coverage",
                "description": "Must use multiple relevant databases",
                "check": self._check_database_coverage,
                "level": ValidationLevel.WARNING
            },
            {
                "rule": "keyword_quality",
                "description": "Keywords must be comprehensive and specific",
                "check": self._check_keyword_quality,
                "level": ValidationLevel.INFO
            },
            {
                "rule": "inclusion_criteria",
                "description": "Inclusion criteria must be clear and specific",
                "check": self._check_inclusion_criteria,
                "level": ValidationLevel.CRITICAL
            },
            {
                "rule": "time_range",
                "description": "Must specify appropriate time range",
                "check": self._check_time_range_appropriateness,
                "level": ValidationLevel.WARNING
            }
        ]
    
    def _load_research_question_rules(self) -> List[Dict]:
        """Load research question validation rules"""
        return [
            {
                "rule": "question_clarity",
                "description": "Research questions must be clear and focused",
                "check": self._check_question_clarity,
                "level": ValidationLevel.CRITICAL
            },
            {
                "rule": "question_feasibility",
                "description": "Research questions must be feasible",
                "check": self._check_question_feasibility,
                "level": ValidationLevel.WARNING
            },
            {
                "rule": "question_relevance",
                "description": "Research questions must be relevant to the field",
                "check": self._check_question_relevance,
                "level": ValidationLevel.INFO
            }
        ]
    
    def _load_ethical_guidelines(self) -> List[Dict]:
        """Load ethical validation rules"""
        return [
            {
                "rule": "informed_consent",
                "description": "Must address informed consent",
                "check": self._check_informed_consent,
                "level": ValidationLevel.CRITICAL
            },
            {
                "rule": "confidentiality",
                "description": "Must address confidentiality and data protection",
                "check": self._check_confidentiality,
                "level": ValidationLevel.CRITICAL
            },
            {
                "rule": "ethical_approval",
                "description": "Must mention ethical approval requirements",
                "check": self._check_ethical_approval,
                "level": ValidationLevel.WARNING
            }
        ]
    
    def _load_quality_metrics(self) -> Dict[str, Dict]:
        """Load quality assessment metrics"""
        return {
            "completeness": {
                "weight": 0.3,
                "criteria": [
                    "research_context_complete",
                    "methodology_complete",
                    "timeline_complete",
                    "outcomes_complete"
                ]
            },
            "methodological_rigor": {
                "weight": 0.4,
                "criteria": [
                    "appropriate_methodology",
                    "valid_data_collection",
                    "suitable_analysis",
                    "ethical_considerations"
                ]
            },
            "feasibility": {
                "weight": 0.2,
                "criteria": [
                    "realistic_timeline",
                    "adequate_resources",
                    "achievable_scope",
                    "practical_constraints"
                ]
            },
            "clarity": {
                "weight": 0.1,
                "criteria": [
                    "clear_objectives",
                    "well_defined_questions",
                    "specific_criteria",
                    "understandable_language"
                ]
            }
        }
    
    def validate_project(self, project_data: Dict) -> ValidationReport:
        """
        Perform comprehensive validation of a research project
        
        Args:
            project_data: Dictionary containing project data
            
        Returns:
            Complete validation report
        """
        logger.info("Starting comprehensive project validation")
        
        report = ValidationReport()
        
        # Validate methodology
        methodology_issues = self._validate_methodology(project_data)
        report.issues.extend(methodology_issues)
        
        # Validate literature search
        literature_issues = self._validate_literature_search(project_data)
        report.issues.extend(literature_issues)
        
        # Validate research questions
        question_issues = self._validate_research_questions(project_data)
        report.issues.extend(question_issues)
        
        # Validate ethical considerations
        ethical_issues = self._validate_ethics(project_data)
        report.issues.extend(ethical_issues)
        
        # Validate completeness
        completeness_issues = self._validate_completeness(project_data)
        report.issues.extend(completeness_issues)
        
        # Validate feasibility
        feasibility_issues = self._validate_feasibility(project_data)
        report.issues.extend(feasibility_issues)
        
        # Calculate overall score
        report.overall_score = self._calculate_quality_score(report.issues)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report.issues)
        
        # Identify enhancement opportunities
        report.enhancement_opportunities = self._identify_enhancement_opportunities(project_data, report.issues)
        
        logger.info(f"Validation completed. Score: {report.overall_score:.2f}")
        return report
    
    def _validate_methodology(self, project_data: Dict) -> List[ValidationIssue]:
        """Validate methodology section"""
        issues = []
        research_type = project_data.get("context", {}).get("research_type", "")
        methodology = project_data.get("methodology", {})
        
        # Get relevant rules for this research type
        rules = self.methodology_rules.get(research_type, [])
        
        for rule in rules:
            try:
                result = rule["check"](methodology, project_data)
                if not result["passed"]:
                    issue = ValidationIssue(
                        category=ValidationCategory.METHODOLOGY,
                        level=rule["level"],
                        title=f"Methodology: {rule['description']}",
                        description=result.get("description", rule["description"]),
                        suggestion=result.get("suggestion", "Review and improve methodology"),
                        auto_fixable=result.get("auto_fixable", False)
                    )
                    issues.append(issue)
            except Exception as e:
                logger.warning(f"Error checking methodology rule {rule['rule']}: {e}")
        
        return issues
    
    def _validate_literature_search(self, project_data: Dict) -> List[ValidationIssue]:
        """Validate literature search strategy"""
        issues = []
        literature_search = project_data.get("literature_search", {})
        
        for rule in self.literature_search_rules:
            try:
                result = rule["check"](literature_search, project_data)
                if not result["passed"]:
                    issue = ValidationIssue(
                        category=ValidationCategory.LITERATURE_SEARCH,
                        level=rule["level"],
                        title=f"Literature Search: {rule['description']}",
                        description=result.get("description", rule["description"]),
                        suggestion=result.get("suggestion", "Improve literature search strategy"),
                        auto_fixable=result.get("auto_fixable", False)
                    )
                    issues.append(issue)
            except Exception as e:
                logger.warning(f"Error checking literature search rule {rule['rule']}: {e}")
        
        return issues
    
    def _validate_research_questions(self, project_data: Dict) -> List[ValidationIssue]:
        """Validate research questions"""
        issues = []
        research_questions = project_data.get("research_questions", [])
        
        for rule in self.research_question_rules:
            try:
                result = rule["check"](research_questions, project_data)
                if not result["passed"]:
                    issue = ValidationIssue(
                        category=ValidationCategory.RESEARCH_QUESTIONS,
                        level=rule["level"],
                        title=f"Research Questions: {rule['description']}",
                        description=result.get("description", rule["description"]),
                        suggestion=result.get("suggestion", "Refine research questions"),
                        auto_fixable=result.get("auto_fixable", False)
                    )
                    issues.append(issue)
            except Exception as e:
                logger.warning(f"Error checking research question rule {rule['rule']}: {e}")
        
        return issues
    
    def _validate_ethics(self, project_data: Dict) -> List[ValidationIssue]:
        """Validate ethical considerations"""
        issues = []
        methodology = project_data.get("methodology", {})
        
        for rule in self.ethical_guidelines:
            try:
                result = rule["check"](methodology, project_data)
                if not result["passed"]:
                    issue = ValidationIssue(
                        category=ValidationCategory.ETHICS,
                        level=rule["level"],
                        title=f"Ethics: {rule['description']}",
                        description=result.get("description", rule["description"]),
                        suggestion=result.get("suggestion", "Address ethical considerations"),
                        auto_fixable=result.get("auto_fixable", False)
                    )
                    issues.append(issue)
            except Exception as e:
                logger.warning(f"Error checking ethical guideline {rule['rule']}: {e}")
        
        return issues
    
    def _validate_completeness(self, project_data: Dict) -> List[ValidationIssue]:
        """Validate project completeness"""
        issues = []
        
        required_sections = [
            ("context", "Research context"),
            ("literature_search", "Literature search strategy"),
            ("methodology", "Research methodology"),
            ("research_questions", "Research questions"),
            ("timeline", "Research timeline"),
            ("expected_outcomes", "Expected outcomes")
        ]
        
        for section_key, section_name in required_sections:
            if not project_data.get(section_key):
                issue = ValidationIssue(
                    category=ValidationCategory.COMPLETENESS,
                    level=ValidationLevel.CRITICAL,
                    title=f"Missing Section: {section_name}",
                    description=f"The {section_name} section is missing or empty",
                    suggestion=f"Complete the {section_name} section with relevant information",
                    auto_fixable=False
                )
                issues.append(issue)
        
        return issues
    
    def _validate_feasibility(self, project_data: Dict) -> List[ValidationIssue]:
        """Validate project feasibility"""
        issues = []
        timeline = project_data.get("timeline", {})
        context = project_data.get("context", {})
        
        # Check timeline feasibility
        if timeline:
            total_weeks = self._calculate_total_timeline(timeline)
            if total_weeks > 52:  # More than 1 year
                issue = ValidationIssue(
                    category=ValidationCategory.FEASIBILITY,
                    level=ValidationLevel.WARNING,
                    title="Extended Timeline",
                    description=f"The timeline spans {total_weeks} weeks, which may be too long",
                    suggestion="Consider breaking the project into phases or focusing on a more specific scope",
                    auto_fixable=False
                )
                issues.append(issue)
        
        # Check scope feasibility based on academic level
        academic_level = context.get("academic_level", "")
        research_type = context.get("research_type", "")
        
        if academic_level == "undergraduate" and research_type == "systematic_review":
            issue = ValidationIssue(
                category=ValidationCategory.FEASIBILITY,
                level=ValidationLevel.WARNING,
                title="Challenging Scope",
                description="Systematic reviews may be too ambitious for undergraduate level",
                suggestion="Consider a scoping review or focus on a more specific topic",
                auto_fixable=False
            )
            issues.append(issue)
        
        return issues
    
    # Individual validation check methods
    def _check_prisma_compliance(self, methodology: Dict, project_data: Dict) -> Dict:
        """Check PRISMA compliance for systematic reviews"""
        # Simplified PRISMA check
        required_elements = ["systematic", "search", "selection", "data extraction", "synthesis"]
        methodology_text = str(methodology).lower()
        
        missing_elements = [elem for elem in required_elements if elem not in methodology_text]
        
        return {
            "passed": len(missing_elements) == 0,
            "description": f"Missing PRISMA elements: {', '.join(missing_elements)}" if missing_elements else "PRISMA elements present",
            "suggestion": "Include all required PRISMA elements in methodology",
            "auto_fixable": False
        }
    
    def _check_search_strategy_completeness(self, literature_search: Dict, project_data: Dict) -> Dict:
        """Check search strategy completeness"""
        required_elements = ["databases", "keywords", "inclusion_criteria"]
        missing_elements = [key for key in required_elements if not literature_search.get(key)]
        
        return {
            "passed": len(missing_elements) == 0,
            "description": f"Missing search elements: {', '.join(missing_elements)}" if missing_elements else "Search strategy complete",
            "suggestion": "Complete all elements of search strategy",
            "auto_fixable": False
        }
    
    def _check_quality_assessment(self, methodology: Dict, project_data: Dict) -> Dict:
        """Check if quality assessment is included"""
        methodology_text = str(methodology).lower()
        quality_terms = ["quality", "bias", "risk", "assessment", "evaluation"]
        
        has_quality = any(term in methodology_text for term in quality_terms)
        
        return {
            "passed": has_quality,
            "description": "Quality assessment mentioned" if has_quality else "No quality assessment found",
            "suggestion": "Include quality assessment of included studies",
            "auto_fixable": False
        }
    
    def _check_theoretical_framework(self, methodology: Dict, project_data: Dict) -> Dict:
        """Check theoretical framework for qualitative studies"""
        framework = methodology.get("theoretical_framework", "")
        
        return {
            "passed": bool(framework),
            "description": f"Theoretical framework: {framework}" if framework else "No theoretical framework specified",
            "suggestion": "Specify a clear theoretical framework",
            "auto_fixable": False
        }
    
    def _check_data_collection_methods(self, methodology: Dict, project_data: Dict) -> Dict:
        """Check data collection methods"""
        methods = methodology.get("data_collection_methods", [])
        
        return {
            "passed": len(methods) > 0,
            "description": f"Data collection methods: {', '.join(methods)}" if methods else "No data collection methods",
            "suggestion": "Specify data collection methods",
            "auto_fixable": False
        }
    
    def _check_trustworthiness_criteria(self, methodology: Dict, project_data: Dict) -> Dict:
        """Check trustworthiness criteria for qualitative studies"""
        methodology_text = str(methodology).lower()
        trustworthiness_terms = ["credibility", "transferability", "dependability", "confirmability", "trustworthiness"]
        
        has_trustworthiness = any(term in methodology_text for term in trustworthiness_terms)
        
        return {
            "passed": has_trustworthiness,
            "description": "Trustworthiness criteria addressed" if has_trustworthiness else "No trustworthiness criteria",
            "suggestion": "Address trustworthiness criteria (credibility, transferability, etc.)",
            "auto_fixable": False
        }
    
    def _check_sample_size_justification(self, methodology: Dict, project_data: Dict) -> Dict:
        """Check sample size justification"""
        methodology_text = str(methodology).lower()
        sample_terms = ["sample size", "power analysis", "justification", "adequate"]
        
        has_justification = any(term in methodology_text for term in sample_terms)
        
        return {
            "passed": has_justification,
            "description": "Sample size justified" if has_justification else "No sample size justification",
            "suggestion": "Justify sample size with power analysis or rationale",
            "auto_fixable": False
        }
    
    def _check_statistical_power(self, methodology: Dict, project_data: Dict) -> Dict:
        """Check statistical power consideration"""
        methodology_text = str(methodology).lower()
        power_terms = ["power", "effect size", "alpha", "beta"]
        
        has_power = any(term in methodology_text for term in power_terms)
        
        return {
            "passed": has_power,
            "description": "Statistical power considered" if has_power else "No statistical power consideration",
            "suggestion": "Consider statistical power in sample size determination",
            "auto_fixable": False
        }
    
    def _check_measurement_validity(self, methodology: Dict, project_data: Dict) -> Dict:
        """Check measurement validity"""
        methodology_text = str(methodology).lower()
        validity_terms = ["validity", "reliability", "measurement", "instrument"]
        
        has_validity = any(term in methodology_text for term in validity_terms)
        
        return {
            "passed": has_validity,
            "description": "Measurement validity addressed" if has_validity else "No measurement validity",
            "suggestion": "Address measurement validity and reliability",
            "auto_fixable": False
        }
    
    def _check_database_coverage(self, literature_search: Dict, project_data: Dict) -> Dict:
        """Check database coverage"""
        databases = literature_search.get("databases", [])
        
        return {
            "passed": len(databases) >= 2,
            "description": f"Using {len(databases)} databases" if databases else "No databases specified",
            "suggestion": "Use at least 2-3 relevant databases for comprehensive coverage",
            "auto_fixable": False
        }
    
    def _check_keyword_quality(self, literature_search: Dict, project_data: Dict) -> Dict:
        """Check keyword quality"""
        keywords = literature_search.get("keywords", [])
        
        # Check for keyword variety and specificity
        quality_score = 0
        if len(keywords) >= 5:
            quality_score += 1
        if any(len(keyword.split()) > 1 for keyword in keywords):  # Multi-word keywords
            quality_score += 1
        
        return {
            "passed": quality_score >= 1,
            "description": f"Keyword quality score: {quality_score}/2",
            "suggestion": "Include specific, multi-word keywords and ensure variety",
            "auto_fixable": False
        }
    
    def _check_inclusion_criteria(self, literature_search: Dict, project_data: Dict) -> Dict:
        """Check inclusion criteria"""
        criteria = literature_search.get("inclusion_criteria", [])
        
        return {
            "passed": len(criteria) >= 3,
            "description": f"Has {len(criteria)} inclusion criteria" if criteria else "No inclusion criteria",
            "suggestion": "Include specific inclusion criteria (study type, population, etc.)",
            "auto_fixable": False
        }
    
    def _check_time_range_appropriateness(self, literature_search: Dict, project_data: Dict) -> Dict:
        """Check time range appropriateness"""
        date_range = literature_search.get("date_range")
        
        return {
            "passed": date_range is not None,
            "description": f"Time range: {date_range}" if date_range else "No time range specified",
            "suggestion": "Specify appropriate date range for literature search",
            "auto_fixable": False
        }
    
    def _check_question_clarity(self, research_questions: List[str], project_data: Dict) -> Dict:
        """Check research question clarity"""
        if not research_questions:
            return {
                "passed": False,
                "description": "No research questions",
                "suggestion": "Add clear, focused research questions",
                "auto_fixable": False
            }
        
        # Check for question words and clarity
        unclear_questions = []
        for question in research_questions:
            question_lower = question.lower()
            if not any(word in question_lower for word in ["what", "how", "why", "whether", "to what extent"]):
                unclear_questions.append(question)
        
        return {
            "passed": len(unclear_questions) == 0,
            "description": f"Unclear questions: {len(unclear_questions)}" if unclear_questions else "Questions are clear",
            "suggestion": "Start questions with what/how/why/whether for clarity",
            "auto_fixable": False
        }
    
    def _check_question_feasibility(self, research_questions: List[str], project_data: Dict) -> Dict:
        """Check research question feasibility"""
        # Simple feasibility check based on scope indicators
        broad_terms = ["everything", "all", "every", "comprehensive", "complete"]
        unfeasible_questions = []
        
        for question in research_questions:
            question_lower = question.lower()
            if any(term in question_lower for term in broad_terms):
                unfeasible_questions.append(question)
        
        return {
            "passed": len(unfeasible_questions) == 0,
            "description": f"Potentially unfeasible questions: {len(unfeasible_questions)}" if unfeasible_questions else "Questions appear feasible",
            "suggestion": "Narrow overly broad questions to make them feasible",
            "auto_fixable": False
        }
    
    def _check_question_relevance(self, research_questions: List[str], project_data: Dict) -> Dict:
        """Check research question relevance"""
        topic = project_data.get("context", {}).get("topic", "").lower()
        
        if not topic:
            return {"passed": True, "description": "Cannot check relevance without topic"}
        
        relevant_questions = 0
        for question in research_questions:
            question_words = set(question.lower().split())
            topic_words = set(topic.split())
            overlap = len(question_words.intersection(topic_words))
            if overlap > 0:
                relevant_questions += 1
        
        return {
            "passed": relevant_questions >= len(research_questions) // 2,
            "description": f"Relevant questions: {relevant_questions}/{len(research_questions)}",
            "suggestion": "Ensure questions are directly related to the research topic",
            "auto_fixable": False
        }
    
    def _check_informed_consent(self, methodology: Dict, project_data: Dict) -> Dict:
        """Check informed consent consideration"""
        methodology_text = str(methodology).lower()
        consent_terms = ["informed consent", "consent", "participant consent", "ethical approval"]
        
        has_consent = any(term in methodology_text for term in consent_terms)
        
        return {
            "passed": has_consent,
            "description": "Informed consent addressed" if has_consent else "No informed consent mentioned",
            "suggestion": "Include informed consent procedures",
            "auto_fixable": False
        }
    
    def _check_confidentiality(self, methodology: Dict, project_data: Dict) -> Dict:
        """Check confidentiality consideration"""
        methodology_text = str(methodology).lower()
        confidentiality_terms = ["confidential", "anonymous", "data protection", "privacy", "secure"]
        
        has_confidentiality = any(term in methodology_text for term in confidentiality_terms)
        
        return {
            "passed": has_confidentiality,
            "description": "Confidentiality addressed" if has_confidentiality else "No confidentiality measures",
            "suggestion": "Include confidentiality and data protection measures",
            "auto_fixable": False
        }
    
    def _check_ethical_approval(self, methodology: Dict, project_data: Dict) -> Dict:
        """Check ethical approval consideration"""
        methodology_text = str(methodology).lower()
        approval_terms = ["ethical approval", "irb", "ethics committee", "institutional review"]
        
        has_approval = any(term in methodology_text for term in approval_terms)
        
        return {
            "passed": has_approval,
            "description": "Ethical approval mentioned" if has_approval else "No ethical approval mentioned",
            "suggestion": "Mention ethical approval requirements",
            "auto_fixable": False
        }
    
    def _calculate_total_timeline(self, timeline: Dict) -> int:
        """Calculate total timeline in weeks"""
        total_weeks = 0
        for phase, duration in timeline.items():
            # Extract number from duration string (e.g., "4-6 weeks" -> 5)
            numbers = re.findall(r'\d+', duration)
            if numbers:
                if len(numbers) == 2:
                    total_weeks += (int(numbers[0]) + int(numbers[1])) // 2
                else:
                    total_weeks += int(numbers[0])
        return total_weeks
    
    def _calculate_quality_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate overall quality score based on issues"""
        # Start with perfect score and deduct based on issues
        score = 1.0
        
        # Deduct points based on issue severity
        for issue in issues:
            if issue.level == ValidationLevel.CRITICAL:
                score -= 0.15
            elif issue.level == ValidationLevel.WARNING:
                score -= 0.08
            elif issue.level == ValidationLevel.INFO:
                score -= 0.03
            elif issue.level == ValidationLevel.SUGGESTION:
                score -= 0.01
        
        return max(0.0, score)
    
    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate recommendations based on validation issues"""
        recommendations = []
        
        # Group issues by category
        critical_issues = [i for i in issues if i.level == ValidationLevel.CRITICAL]
        warning_issues = [i for i in issues if i.level == ValidationLevel.WARNING]
        
        if critical_issues:
            recommendations.append("Address all critical issues before proceeding with the research")
        
        if len(critical_issues) > 3:
            recommendations.append("Consider simplifying the research scope to address critical issues")
        
        # Category-specific recommendations
        methodology_issues = [i for i in issues if i.category == ValidationCategory.METHODOLOGY]
        if methodology_issues:
            recommendations.append("Review and strengthen the research methodology")
        
        literature_issues = [i for i in issues if i.category == ValidationCategory.LITERATURE_SEARCH]
        if literature_issues:
            recommendations.append("Improve literature search strategy for comprehensive coverage")
        
        ethical_issues = [i for i in issues if i.category == ValidationCategory.ETHICS]
        if ethical_issues:
            recommendations.append("Ensure all ethical considerations are properly addressed")
        
        return recommendations
    
    def _identify_enhancement_opportunities(self, project_data: Dict, issues: List[ValidationIssue]) -> List[str]:
        """Identify opportunities for project enhancement"""
        opportunities = []
        
        # Check for enhancement opportunities based on project characteristics
        research_type = project_data.get("context", {}).get("research_type", "")
        
        if research_type == "systematic_review":
            opportunities.append("Consider registering the systematic review protocol in PROSPERO")
            opportunities.append("Plan for risk of bias assessment using standardized tools")
        
        if research_type == "qualitative_study":
            opportunities.append("Consider member checking to enhance credibility")
            opportunities.append("Plan for reflexive journaling throughout the research process")
        
        # Check for innovative approaches
        methodology = project_data.get("methodology", {})
        if not any("mixed" in str(methodology).lower() for _ in [1]):
            opportunities.append("Consider mixed methods approach for comprehensive insights")
        
        # Check for technology integration
        opportunities.append("Explore digital tools for data collection and analysis")
        opportunities.append("Consider open science practices for transparency and reproducibility")
        
        return opportunities
    
    def export_report(self, report: ValidationReport, format: str = "json") -> str:
        """Export validation report in specified format"""
        if format.lower() == "json":
            return json.dumps(report.__dict__, indent=2, default=str)
        elif format.lower() == "markdown":
            return self._export_markdown(report)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_markdown(self, report: ValidationReport) -> str:
        """Export validation report as markdown"""
        md = f"""# Research Project Validation Report

**Overall Quality Score**: {report.overall_score:.2f}/1.00  
**Validated**: {report.validated_at}

## Summary
- **Total Issues**: {len(report.issues)}
- **Critical Issues**: {len([i for i in report.issues if i.level == ValidationLevel.CRITICAL])}
- **Warnings**: {len([i for i in report.issues if i.level == ValidationLevel.WARNING])}
- **Info**: {len([i for i in report.issues if i.level == ValidationLevel.INFO])}
- **Suggestions**: {len([i for i in report.issues if i.level == ValidationLevel.SUGGESTION])}

## Issues by Severity

### 🚨 Critical Issues
"""
        
        critical_issues = [i for i in report.issues if i.level == ValidationLevel.CRITICAL]
        for issue in critical_issues:
            md += f"""
#### {issue.title}
**Category**: {issue.category.value}  
**Description**: {issue.description}  
**Suggestion**: {issue.suggestion}
"""
        
        md += """
### ⚠️ Warnings
"""
        
        warning_issues = [i for i in report.issues if i.level == ValidationLevel.WARNING]
        for issue in warning_issues:
            md += f"""
#### {issue.title}
**Category**: {issue.category.value}  
**Description**: {issue.description}  
**Suggestion**: {issue.suggestion}
"""
        
        md += """
### ℹ️ Information
"""
        
        info_issues = [i for i in report.issues if i.level == ValidationLevel.INFO]
        for issue in info_issues:
            md += f"""
#### {issue.title}
**Category**: {issue.category.value}  
**Description**: {issue.description}  
**Suggestion**: {issue.suggestion}
"""
        
        md += f"""
## Recommendations
{chr(10).join(f"- {rec}" for rec in report.recommendations)}

## Enhancement Opportunities
{chr(10).join(f"- {opp}" for opp in report.enhancement_opportunities)}

## Quality Breakdown
"""
        
        # Add quality breakdown
        categories = {}
        for issue in report.issues:
            cat = issue.category.value
            if cat not in categories:
                categories[cat] = {"critical": 0, "warning": 0, "info": 0, "suggestion": 0}
            categories[cat][issue.level.value] += 1
        
        for category, counts in categories.items():
            md += f"\n**{category.replace('_', ' ').title()}**: "
            total = sum(counts.values())
            if total > 0:
                md += f"{total} issues ({counts['critical']} critical, {counts['warning']} warnings)"
            else:
                md += "No issues"
        
        md += "\n---\n*Report generated by AI Research Project Validator*"
        
        return md

def main():
    """Example usage of the Validation Engine"""
    validator = ValidationEngine()
    
    # Example project data (would normally come from the project generator)
    project_data = {
        "context": {
            "research_type": "systematic_review",
            "academic_level": "graduate",
            "discipline": "psychology"
        },
        "methodology": {
            "approach": "Systematic literature review",
            "data_collection_methods": ["Database search", "Screening"],
            "ethical_considerations": ["Informed consent", "Confidentiality"]
        },
        "literature_search": {
            "databases": ["PubMed", "PsycINFO"],
            "keywords": ["remote work", "productivity", "well-being"],
            "inclusion_criteria": ["Peer-reviewed", "English language"]
        },
        "research_questions": [
            "How does remote work affect employee productivity?",
            "What factors influence well-being in remote work?"
        ],
        "timeline": {
            "Literature Review": "4-6 weeks",
            "Data Collection": "8-12 weeks"
        }
    }
    
    # Validate project
    report = validator.validate_project(project_data)
    
    print("=== VALIDATION REPORT ===")
    print(validator.export_report(report, "markdown"))

if __name__ == "__main__":
    main()
