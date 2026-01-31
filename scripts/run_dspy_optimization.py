#!/usr/bin/env python3
"""
DSPy Optimization Script

This script runs DSPy optimization on the research modules to improve
prompt quality through automated prompt engineering.

Usage:
    # Run with default settings (requires OPENAI_API_KEY)
    python scripts/run_dspy_optimization.py

    # Run with Ollama (local LLM)
    python scripts/run_dspy_optimization.py --model ollama_chat/llama3.1:8b

    # Run with specific optimization level
    python scripts/run_dspy_optimization.py --level medium

    # Save optimized modules
    python scripts/run_dspy_optimization.py --save-path ./optimized_modules/
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dspy
from loguru import logger

from ai_research_generator.optimization.dspy_modules import (
    TopicAnalyzerModule,
    PaperSynthesizerModule,
    MethodologyRecommenderModule,
    optimize_topic_analyzer,
    optimize_paper_synthesizer,
    create_topic_analysis_examples,
    create_paper_synthesis_examples,
    topic_analysis_metric,
    paper_synthesis_metric,
)


def setup_dspy(model_name: str) -> None:
    """Configure DSPy with the specified model."""
    logger.info(f"Configuring DSPy with model: {model_name}")

    if model_name.startswith("ollama"):
        # Ollama local model
        lm = dspy.LM(model_name, api_base="http://localhost:11434")
    elif model_name.startswith("openai/"):
        # OpenAI model
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable required for OpenAI models")
        lm = dspy.LM(model_name)
    else:
        # Default to OpenAI-compatible
        lm = dspy.LM(model_name)

    dspy.configure(lm=lm)
    logger.info("DSPy configured successfully")


def test_module(module: dspy.Module, test_input: dict, module_name: str) -> None:
    """Test a module with sample input."""
    logger.info(f"Testing {module_name}...")
    try:
        result = module(**test_input)
        logger.info(f"{module_name} output:")
        for key, value in result.items():
            if isinstance(value, list):
                logger.info(f"  {key}: {value[:3]}..." if len(value) > 3 else f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {str(value)[:100]}...")
    except Exception as e:
        logger.error(f"Error testing {module_name}: {e}")


def run_optimization(
    model_name: str,
    optimization_level: str,
    save_path: str | None,
    modules: list[str],
) -> None:
    """Run DSPy optimization on specified modules."""
    setup_dspy(model_name)

    results = {}

    if "topic" in modules or "all" in modules:
        logger.info("=" * 60)
        logger.info("Optimizing Topic Analyzer Module")
        logger.info("=" * 60)

        trainset = create_topic_analysis_examples()
        logger.info(f"Training set size: {len(trainset)} examples")

        try:
            optimized_topic = optimize_topic_analyzer(
                trainset=trainset,
                model_name=model_name,
                save_path=f"{save_path}/topic_analyzer.json" if save_path else None,
                optimization_level=optimization_level,
            )
            results["topic_analyzer"] = optimized_topic

            # Test the optimized module
            test_module(
                optimized_topic,
                {
                    "topic": "Quantum computing applications in cryptography",
                    "research_question": "How can quantum computers break current encryption?",
                },
                "Optimized Topic Analyzer",
            )
        except Exception as e:
            logger.error(f"Topic analyzer optimization failed: {e}")

    if "synthesis" in modules or "all" in modules:
        logger.info("=" * 60)
        logger.info("Optimizing Paper Synthesizer Module")
        logger.info("=" * 60)

        trainset = create_paper_synthesis_examples()
        logger.info(f"Training set size: {len(trainset)} examples")

        try:
            optimized_synthesis = optimize_paper_synthesizer(
                trainset=trainset,
                model_name=model_name,
                save_path=f"{save_path}/paper_synthesizer.json" if save_path else None,
                optimization_level=optimization_level,
            )
            results["paper_synthesizer"] = optimized_synthesis

            # Test the optimized module
            test_module(
                optimized_synthesis,
                {
                    "topic": "Machine learning in finance",
                    "papers_summary": """
                    1. "Deep Learning for Stock Prediction" (2024) - Uses LSTM for price forecasting
                    2. "AI Risk Management" (2023) - ML models for credit risk assessment
                    3. "Algorithmic Trading Review" (2024) - Survey of ML trading strategies
                    """,
                },
                "Optimized Paper Synthesizer",
            )
        except Exception as e:
            logger.error(f"Paper synthesizer optimization failed: {e}")

    logger.info("=" * 60)
    logger.info("Optimization Complete")
    logger.info("=" * 60)
    logger.info(f"Optimized modules: {list(results.keys())}")

    if save_path:
        logger.info(f"Saved to: {save_path}")


def run_evaluation_only(model_name: str) -> None:
    """Run evaluation on unoptimized modules to establish baseline."""
    setup_dspy(model_name)

    logger.info("=" * 60)
    logger.info("Running Baseline Evaluation (No Optimization)")
    logger.info("=" * 60)

    # Test Topic Analyzer
    topic_module = TopicAnalyzerModule()
    test_cases = create_topic_analysis_examples()

    scores = []
    for example in test_cases:
        try:
            prediction = topic_module(
                topic=example.topic,
                research_question=example.research_question,
            )
            score = topic_analysis_metric(example, prediction)
            scores.append(score)
            logger.info(f"Topic: {example.topic[:40]}... Score: {score:.2f}")
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0
    logger.info(f"Average Topic Analyzer Score: {avg_score:.2f}")

    # Test Paper Synthesizer
    synthesis_module = PaperSynthesizerModule()
    synthesis_cases = create_paper_synthesis_examples()

    synthesis_scores = []
    for example in synthesis_cases:
        try:
            prediction = synthesis_module(
                topic=example.topic,
                papers_summary=example.papers_summary,
            )
            score = paper_synthesis_metric(example, prediction)
            synthesis_scores.append(score)
            logger.info(f"Topic: {example.topic[:40]}... Score: {score:.2f}")
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            synthesis_scores.append(0.0)

    avg_synthesis = sum(synthesis_scores) / len(synthesis_scores) if synthesis_scores else 0
    logger.info(f"Average Paper Synthesizer Score: {avg_synthesis:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run DSPy optimization on research modules")
    parser.add_argument(
        "--model",
        default="ollama_chat/llama3.1:8b",
        help="Model to use (e.g., 'openai/gpt-4o-mini', 'ollama_chat/llama3.1:8b')",
    )
    parser.add_argument(
        "--level",
        choices=["light", "medium", "heavy"],
        default="light",
        help="Optimization level (light=fast, heavy=thorough)",
    )
    parser.add_argument(
        "--save-path",
        help="Directory to save optimized modules",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=["topic", "synthesis", "all"],
        default=["all"],
        help="Which modules to optimize",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation without optimization",
    )

    args = parser.parse_args()

    # Create save directory if needed
    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        run_evaluation_only(args.model)
    else:
        run_optimization(
            model_name=args.model,
            optimization_level=args.level,
            save_path=args.save_path,
            modules=args.modules,
        )


if __name__ == "__main__":
    main()
