"""
DSPy modules for offline prompt optimization.

This module provides DSPy-based modules that can be optimized offline
using MIPROv2 or BootstrapFewShot optimizers, then saved and loaded
for production use.
"""

import json
from pathlib import Path
from typing import Optional, Any

import dspy
from loguru import logger


# =============================================================================
# DSPy Signatures (Define Input/Output Schema)
# =============================================================================

class TopicAnalysisSignature(dspy.Signature):
    """Analyze a research topic and extract key concepts, scope, and complexity."""
    
    topic: str = dspy.InputField(desc="The research topic to analyze")
    research_question: str = dspy.InputField(desc="The main research question")
    
    key_concepts: list[str] = dspy.OutputField(desc="Key concepts and terms (5-10 items)")
    research_scope: str = dspy.OutputField(desc="Scope: narrow, moderate, or broad")
    complexity_level: str = dspy.OutputField(desc="Complexity: basic, intermediate, or advanced")
    suggested_subtopics: list[str] = dspy.OutputField(desc="Suggested subtopics for investigation")
    potential_challenges: list[str] = dspy.OutputField(desc="Potential research challenges")


class PaperSynthesisSignature(dspy.Signature):
    """Synthesize findings from academic papers into themes and insights."""
    
    topic: str = dspy.InputField(desc="The research topic")
    papers_summary: str = dspy.InputField(desc="Summary of papers with titles and abstracts")
    
    main_themes: list[str] = dspy.OutputField(desc="Main themes identified across papers")
    research_gaps: list[str] = dspy.OutputField(desc="Gaps in current research")
    key_findings: list[str] = dspy.OutputField(desc="Key findings from the literature")
    synthesis_narrative: str = dspy.OutputField(desc="Narrative synthesis of the literature")


class MethodologySignature(dspy.Signature):
    """Recommend appropriate research methodology based on context."""
    
    topic: str = dspy.InputField(desc="The research topic")
    research_type: str = dspy.InputField(desc="Type of research (e.g., systematic_review)")
    discipline: str = dspy.InputField(desc="Academic discipline")
    academic_level: str = dspy.InputField(desc="Academic level (undergraduate, graduate, doctoral)")
    
    primary_methodology: str = dspy.OutputField(desc="Primary recommended methodology")
    rationale: str = dspy.OutputField(desc="Rationale for the recommendation")
    data_collection_methods: list[str] = dspy.OutputField(desc="Recommended data collection methods")
    analysis_techniques: list[str] = dspy.OutputField(desc="Recommended analysis techniques")


# =============================================================================
# DSPy Modules (Composable Units)
# =============================================================================

class TopicAnalyzerModule(dspy.Module):
    """DSPy module for topic analysis with chain-of-thought reasoning."""
    
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(TopicAnalysisSignature)
    
    def forward(self, topic: str, research_question: str) -> dspy.Prediction:
        """Analyze a research topic."""
        return self.analyze(topic=topic, research_question=research_question)


class PaperSynthesizerModule(dspy.Module):
    """DSPy module for paper synthesis with chain-of-thought reasoning."""
    
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(PaperSynthesisSignature)
    
    def forward(self, topic: str, papers_summary: str) -> dspy.Prediction:
        """Synthesize papers into themes and insights."""
        return self.synthesize(topic=topic, papers_summary=papers_summary)


class MethodologyRecommenderModule(dspy.Module):
    """DSPy module for methodology recommendation."""
    
    def __init__(self):
        super().__init__()
        self.recommend = dspy.ChainOfThought(MethodologySignature)
    
    def forward(
        self,
        topic: str,
        research_type: str,
        discipline: str,
        academic_level: str
    ) -> dspy.Prediction:
        """Recommend research methodology."""
        return self.recommend(
            topic=topic,
            research_type=research_type,
            discipline=discipline,
            academic_level=academic_level
        )


# =============================================================================
# Evaluation Metrics for Optimization
# =============================================================================

def topic_analysis_metric(example: dspy.Example, prediction: dspy.Prediction) -> float:
    """
    Metric for evaluating topic analysis quality.
    
    Scores based on:
    - Number of key concepts (target: 5-10)
    - Valid scope value
    - Valid complexity value
    - Presence of subtopics and challenges
    """
    score = 0.0
    
    # Check key concepts (0-40 points)
    concepts = prediction.key_concepts if hasattr(prediction, 'key_concepts') else []
    if isinstance(concepts, list):
        if 5 <= len(concepts) <= 10:
            score += 40
        elif 3 <= len(concepts) <= 12:
            score += 25
        elif len(concepts) > 0:
            score += 10
    
    # Check scope validity (0-20 points)
    scope = getattr(prediction, 'research_scope', '')
    if scope.lower() in ['narrow', 'moderate', 'broad']:
        score += 20
    
    # Check complexity validity (0-20 points)
    complexity = getattr(prediction, 'complexity_level', '')
    if complexity.lower() in ['basic', 'intermediate', 'advanced']:
        score += 20
    
    # Check subtopics (0-10 points)
    subtopics = getattr(prediction, 'suggested_subtopics', [])
    if isinstance(subtopics, list) and len(subtopics) >= 2:
        score += 10
    
    # Check challenges (0-10 points)
    challenges = getattr(prediction, 'potential_challenges', [])
    if isinstance(challenges, list) and len(challenges) >= 1:
        score += 10
    
    return score / 100.0


def paper_synthesis_metric(example: dspy.Example, prediction: dspy.Prediction) -> float:
    """
    Metric for evaluating paper synthesis quality.
    
    Scores based on:
    - Presence of themes
    - Identification of gaps
    - Key findings extracted
    - Narrative quality (length)
    """
    score = 0.0
    
    # Check themes (0-30 points)
    themes = getattr(prediction, 'main_themes', [])
    if isinstance(themes, list) and len(themes) >= 2:
        score += 30
    elif isinstance(themes, list) and len(themes) >= 1:
        score += 15
    
    # Check gaps (0-25 points)
    gaps = getattr(prediction, 'research_gaps', [])
    if isinstance(gaps, list) and len(gaps) >= 2:
        score += 25
    elif isinstance(gaps, list) and len(gaps) >= 1:
        score += 12
    
    # Check findings (0-25 points)
    findings = getattr(prediction, 'key_findings', [])
    if isinstance(findings, list) and len(findings) >= 3:
        score += 25
    elif isinstance(findings, list) and len(findings) >= 1:
        score += 12
    
    # Check narrative (0-20 points)
    narrative = getattr(prediction, 'synthesis_narrative', '')
    if isinstance(narrative, str) and len(narrative) >= 100:
        score += 20
    elif isinstance(narrative, str) and len(narrative) >= 50:
        score += 10
    
    return score / 100.0


# =============================================================================
# Optimization Functions
# =============================================================================

def optimize_topic_analyzer(
    trainset: list[dspy.Example],
    model_name: str = "openai/gpt-4o-mini",
    save_path: Optional[str] = None,
    optimization_level: str = "light"
) -> TopicAnalyzerModule:
    """
    Optimize the topic analyzer module using MIPROv2.
    
    Args:
        trainset: List of training examples with topic, research_question
        model_name: LLM model to use for optimization
        save_path: Path to save the optimized module
        optimization_level: "light", "medium", or "heavy"
        
    Returns:
        Optimized TopicAnalyzerModule
    """
    logger.info(f"Optimizing topic analyzer with {len(trainset)} examples")
    
    # Configure DSPy
    lm = dspy.LM(model_name)
    dspy.configure(lm=lm)
    
    # Create module
    module = TopicAnalyzerModule()
    
    # Create optimizer
    optimizer = dspy.MIPROv2(
        metric=topic_analysis_metric,
        auto=optimization_level,
        num_threads=4
    )
    
    # Optimize
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        max_bootstrapped_demos=2,
        max_labeled_demos=2
    )
    
    # Save if path provided
    if save_path:
        optimized.save(save_path)
        logger.info(f"Saved optimized module to {save_path}")
    
    return optimized


def optimize_paper_synthesizer(
    trainset: list[dspy.Example],
    model_name: str = "openai/gpt-4o-mini",
    save_path: Optional[str] = None,
    optimization_level: str = "light"
) -> PaperSynthesizerModule:
    """
    Optimize the paper synthesizer module using MIPROv2.
    
    Args:
        trainset: List of training examples
        model_name: LLM model to use
        save_path: Path to save the optimized module
        optimization_level: "light", "medium", or "heavy"
        
    Returns:
        Optimized PaperSynthesizerModule
    """
    logger.info(f"Optimizing paper synthesizer with {len(trainset)} examples")
    
    # Configure DSPy
    lm = dspy.LM(model_name)
    dspy.configure(lm=lm)
    
    # Create module
    module = PaperSynthesizerModule()
    
    # Create optimizer
    optimizer = dspy.MIPROv2(
        metric=paper_synthesis_metric,
        auto=optimization_level,
        num_threads=4
    )
    
    # Optimize
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        max_bootstrapped_demos=2,
        max_labeled_demos=2
    )
    
    # Save if path provided
    if save_path:
        optimized.save(save_path)
        logger.info(f"Saved optimized module to {save_path}")
    
    return optimized


def load_optimized_module(
    module_class: type,
    path: str
) -> dspy.Module:
    """
    Load a previously optimized DSPy module.
    
    Args:
        module_class: The module class (e.g., TopicAnalyzerModule)
        path: Path to the saved module
        
    Returns:
        Loaded module with optimized prompts
    """
    module = module_class()
    module.load(path)
    logger.info(f"Loaded optimized module from {path}")
    return module


# =============================================================================
# Training Data Generation
# =============================================================================

def create_topic_analysis_examples() -> list[dspy.Example]:
    """
    Create sample training examples for topic analysis optimization.
    
    In production, these would come from human-labeled data or
    high-quality LLM-generated examples.
    """
    examples = [
        dspy.Example(
            topic="Impact of artificial intelligence on healthcare diagnostics",
            research_question="How does AI improve diagnostic accuracy in radiology?"
        ).with_inputs("topic", "research_question"),
        
        dspy.Example(
            topic="Climate change effects on marine biodiversity",
            research_question="What are the primary impacts of ocean warming on coral reef ecosystems?"
        ).with_inputs("topic", "research_question"),
        
        dspy.Example(
            topic="Remote work productivity and employee wellbeing",
            research_question="How does remote work affect employee mental health and productivity?"
        ).with_inputs("topic", "research_question"),
        
        dspy.Example(
            topic="Blockchain technology in supply chain management",
            research_question="Can blockchain improve transparency and traceability in global supply chains?"
        ).with_inputs("topic", "research_question"),
        
        dspy.Example(
            topic="Machine learning in financial fraud detection",
            research_question="How effective are ML models at detecting fraudulent transactions in real-time?"
        ).with_inputs("topic", "research_question"),
    ]
    
    return examples


def create_paper_synthesis_examples() -> list[dspy.Example]:
    """
    Create sample training examples for paper synthesis optimization.
    """
    examples = [
        dspy.Example(
            topic="AI in healthcare",
            papers_summary="""
            1. "Deep Learning for Medical Imaging" (2023) - Reviews CNN applications in radiology
            2. "AI-Assisted Diagnosis" (2024) - Compares AI vs human diagnostic accuracy
            3. "Challenges in Clinical AI" (2023) - Discusses regulatory and ethical barriers
            """
        ).with_inputs("topic", "papers_summary"),
        
        dspy.Example(
            topic="Sustainable energy",
            papers_summary="""
            1. "Solar Panel Efficiency Trends" (2024) - Analyzes improvements in PV technology
            2. "Grid Integration Challenges" (2023) - Examines renewable energy storage
            3. "Policy Impacts on Adoption" (2024) - Studies government incentive effects
            """
        ).with_inputs("topic", "papers_summary"),
    ]
    
    return examples
