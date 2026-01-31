#!/usr/bin/env python3
"""
DSPy Prompt Optimization CLI.

This script runs DSPy optimization on research workflow modules
to find optimal prompts and few-shot examples.

Usage:
    python -m app.optimization.run_optimization --level light --module all
    python -m app.optimization.run_optimization --level medium --module topic_analyzer
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import dspy
from loguru import logger

from app.optimization.dspy_modules import (
    TopicAnalyzerModule,
    PaperSynthesizerModule,
    topic_analysis_metric,
    paper_synthesis_metric,
    create_topic_analysis_examples,
    create_paper_synthesis_examples,
)


# Output directory for optimized modules
OUTPUT_DIR = Path("optimized_modules")


def setup_dspy(model: str) -> None:
    """Configure DSPy with the specified model."""
    logger.info(f"Configuring DSPy with model: {model}")
    lm = dspy.LM(model)
    dspy.configure(lm=lm)


def optimize_topic_analyzer(level: str, model: str, output_dir: Path) -> dict:
    """Optimize the topic analyzer module."""
    logger.info("Optimizing TopicAnalyzerModule...")

    # Get training examples
    trainset = create_topic_analysis_examples()
    logger.info(f"Using {len(trainset)} training examples")

    # Create module
    module = TopicAnalyzerModule()

    # Configure optimizer based on level
    num_threads = {"light": 4, "medium": 8, "heavy": 16}.get(level, 4)

    optimizer = dspy.MIPROv2(metric=topic_analysis_metric, auto=level, num_threads=num_threads)

    # Run optimization
    logger.info(f"Running {level} optimization...")
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        max_bootstrapped_demos=2 if level == "light" else 4,
        max_labeled_demos=2 if level == "light" else 4,
    )

    # Save optimized module
    output_path = output_dir / "topic_analyzer.json"
    optimized.save(str(output_path))
    logger.info(f"Saved optimized module to {output_path}")

    return {
        "module": "topic_analyzer",
        "status": "success",
        "output_path": str(output_path),
        "training_examples": len(trainset),
        "optimization_level": level,
    }


def optimize_paper_synthesizer(level: str, model: str, output_dir: Path) -> dict:
    """Optimize the paper synthesizer module."""
    logger.info("Optimizing PaperSynthesizerModule...")

    # Get training examples
    trainset = create_paper_synthesis_examples()
    logger.info(f"Using {len(trainset)} training examples")

    # Create module
    module = PaperSynthesizerModule()

    # Configure optimizer
    num_threads = {"light": 4, "medium": 8, "heavy": 16}.get(level, 4)

    optimizer = dspy.MIPROv2(metric=paper_synthesis_metric, auto=level, num_threads=num_threads)

    # Run optimization
    logger.info(f"Running {level} optimization...")
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        max_bootstrapped_demos=2 if level == "light" else 4,
        max_labeled_demos=2 if level == "light" else 4,
    )

    # Save optimized module
    output_path = output_dir / "paper_synthesizer.json"
    optimized.save(str(output_path))
    logger.info(f"Saved optimized module to {output_path}")

    return {
        "module": "paper_synthesizer",
        "status": "success",
        "output_path": str(output_path),
        "training_examples": len(trainset),
        "optimization_level": level,
    }


def optimize_methodology_recommender(level: str, model: str, output_dir: Path) -> dict:
    """Optimize the methodology recommender module."""
    logger.info("Optimizing MethodologyRecommenderModule...")

    # For now, methodology recommender uses rule-based logic
    # This is a placeholder for future optimization
    logger.warning("MethodologyRecommenderModule optimization not yet implemented")

    return {
        "module": "methodology_recommender",
        "status": "skipped",
        "reason": "Rule-based module, no LLM optimization needed",
    }


def main():
    parser = argparse.ArgumentParser(
        description="DSPy Prompt Optimization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Light optimization of all modules
  python -m app.optimization.run_optimization --level light --module all
  
  # Medium optimization of topic analyzer only
  python -m app.optimization.run_optimization --level medium --module topic_analyzer
  
  # Heavy optimization with custom model
  python -m app.optimization.run_optimization --level heavy --model openai/gpt-4o
        """,
    )

    parser.add_argument(
        "--level",
        choices=["light", "medium", "heavy"],
        default="light",
        help="Optimization level (light=faster, heavy=better quality)",
    )

    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="LLM model for optimization (e.g., openai/gpt-4o-mini, anthropic/claude-3-haiku)",
    )

    parser.add_argument(
        "--module",
        choices=["all", "topic_analyzer", "paper_synthesizer", "methodology_recommender"],
        default="all",
        help="Which module to optimize",
    )

    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Directory to save optimized modules"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running optimization",
    )

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("=" * 60)
    logger.info("DSPy Prompt Optimization")
    logger.info("=" * 60)
    logger.info(f"Level: {args.level}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Module: {args.module}")
    logger.info(f"Output: {args.output_dir}")

    if args.dry_run:
        logger.info("DRY RUN - No optimization will be performed")
        return 0

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup DSPy
    try:
        setup_dspy(args.model)
    except Exception as e:
        logger.error(f"Failed to configure DSPy: {e}")
        logger.error("Make sure OPENAI_API_KEY or appropriate API key is set")
        return 1

    # Run optimization
    results = []

    try:
        if args.module in ["all", "topic_analyzer"]:
            result = optimize_topic_analyzer(args.level, args.model, args.output_dir)
            results.append(result)

        if args.module in ["all", "paper_synthesizer"]:
            result = optimize_paper_synthesizer(args.level, args.model, args.output_dir)
            results.append(result)

        if args.module in ["all", "methodology_recommender"]:
            result = optimize_methodology_recommender(args.level, args.model, args.output_dir)
            results.append(result)

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1

    # Save results summary
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": args.model,
        "level": args.level,
        "results": results,
    }

    summary_path = args.output_dir / "optimization_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("Optimization Complete")
    logger.info("=" * 60)

    for result in results:
        status = "✅" if result["status"] == "success" else "⏭️"
        logger.info(f"{status} {result['module']}: {result['status']}")

    logger.info(f"Summary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
