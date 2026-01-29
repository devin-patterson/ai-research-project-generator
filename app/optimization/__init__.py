"""DSPy-based prompt optimization for research workflows."""

from app.optimization.dspy_modules import (
    TopicAnalyzerModule,
    PaperSynthesizerModule,
    MethodologyRecommenderModule,
    optimize_topic_analyzer,
    optimize_paper_synthesizer,
    load_optimized_module,
)

__all__ = [
    "TopicAnalyzerModule",
    "PaperSynthesizerModule",
    "MethodologyRecommenderModule",
    "optimize_topic_analyzer",
    "optimize_paper_synthesizer",
    "load_optimized_module",
]
