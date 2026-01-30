#!/usr/bin/env python3
"""
LLM Provider Module

Provides integration with local LLM models via Ollama and OpenAI-compatible APIs.
Supports multiple model backends for research project generation.

Author: AI Research Assistant
Date: 2025-01-29
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Generator, Any
from enum import Enum

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

import httpx


class LLMProvider(Enum):
    """Supported LLM providers"""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL_OPENAI_COMPATIBLE = "local_openai_compatible"


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""

    provider: LLMProvider
    model: str
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create config from environment variables"""
        provider = LLMProvider(os.getenv("LLM_PROVIDER", "ollama"))

        defaults = {
            LLMProvider.OLLAMA: {"model": "llama3.1:8b", "base_url": "http://localhost:11434"},
            LLMProvider.OPENAI: {"model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
            LLMProvider.ANTHROPIC: {
                "model": "claude-3-5-sonnet-20241022",
                "base_url": "https://api.anthropic.com",
            },
            LLMProvider.LOCAL_OPENAI_COMPATIBLE: {
                "model": "local-model",
                "base_url": "http://localhost:8080/v1",
            },
        }

        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", defaults[provider]["model"]),
            base_url=os.getenv("LLM_BASE_URL", defaults[provider]["base_url"]),
            api_key=os.getenv("LLM_API_KEY"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            timeout=int(os.getenv("LLM_TIMEOUT", "120")),
        )


@dataclass
class LLMResponse:
    """Response from LLM"""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


class BaseLLMClient(ABC):
    """Base class for LLM clients"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = httpx.Client(timeout=config.timeout)

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response from the LLM"""
        pass

    @abstractmethod
    def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM"""
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Chat with the LLM using message history"""
        pass

    def close(self):
        """Close the HTTP client"""
        self.client.close()


class OllamaClient(BaseLLMClient):
    """Client for Ollama local LLM server"""

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response using Ollama"""
        url = f"{self.config.base_url}/api/generate"

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data.get("response", ""),
                model=data.get("model", self.config.model),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
                finish_reason=data.get("done_reason", "stop"),
            )
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Generate a streaming response using Ollama"""
        url = f"{self.config.base_url}/api/generate"

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
        except httpx.HTTPError as e:
            logger.error(f"Ollama streaming error: {e}")
            raise

    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Chat with Ollama using message history"""
        url = f"{self.config.base_url}/api/chat"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        try:
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", self.config.model),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
                finish_reason=data.get("done_reason", "stop"),
            )
        except httpx.HTTPError as e:
            logger.error(f"Ollama chat error: {e}")
            raise

    def list_models(self) -> List[str]:
        """List available models in Ollama"""
        url = f"{self.config.base_url}/api/tags"

        try:
            response = self.client.get(url)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except httpx.HTTPError as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        url = f"{self.config.base_url}/api/pull"

        try:
            with self.client.stream("POST", url, json={"name": model_name}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        logger.info(f"Pulling {model_name}: {status}")
            return True
        except httpx.HTTPError as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False


class OpenAICompatibleClient(BaseLLMClient):
    """Client for OpenAI-compatible APIs (OpenAI, local servers like LM Studio, vLLM)"""

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response using OpenAI-compatible API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat(messages)

    def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Generate a streaming response using OpenAI-compatible API"""
        url = f"{self.config.base_url}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }

        try:
            with self.client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPError as e:
            logger.error(f"OpenAI-compatible streaming error: {e}")
            raise

    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Chat using OpenAI-compatible API"""
        url = f"{self.config.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            response = self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]

            return LLMResponse(
                content=choice.get("message", {}).get("content", ""),
                model=data.get("model", self.config.model),
                usage=data.get("usage"),
                finish_reason=choice.get("finish_reason"),
            )
        except httpx.HTTPError as e:
            logger.error(f"OpenAI-compatible API error: {e}")
            raise


def create_llm_client(config: Optional[LLMConfig] = None) -> BaseLLMClient:
    """Factory function to create appropriate LLM client"""
    if config is None:
        config = LLMConfig.from_env()

    if config.provider == LLMProvider.OLLAMA:
        return OllamaClient(config)
    elif config.provider in [LLMProvider.OPENAI, LLMProvider.LOCAL_OPENAI_COMPATIBLE]:
        return OpenAICompatibleClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")


# Research-specific prompts
RESEARCH_SYSTEM_PROMPT = """You are an expert academic research assistant with deep knowledge of research methodologies, 
systematic reviews, and academic writing. You help researchers design rigorous research projects by:

1. Analyzing research topics to identify key concepts, variables, and relationships
2. Recommending appropriate methodologies based on research questions and objectives
3. Identifying relevant theoretical frameworks and literature
4. Suggesting comprehensive search strategies for literature reviews
5. Evaluating research designs for methodological rigor
6. Identifying potential limitations and ethical considerations

Always provide evidence-based recommendations and cite relevant methodological literature when appropriate.
Be specific, detailed, and academically rigorous in your responses."""


class ResearchLLMAssistant:
    """LLM-powered research assistant for enhanced project generation"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.client = create_llm_client(config)
        self.system_prompt = RESEARCH_SYSTEM_PROMPT

    def analyze_topic(self, topic: str, discipline: str) -> Dict[str, Any]:
        """Analyze a research topic using LLM"""
        prompt = f"""Analyze the following research topic in the field of {discipline}:

Topic: {topic}

Provide a comprehensive analysis including:
1. Key concepts and their definitions
2. Important variables to consider
3. Potential theoretical frameworks
4. Related research areas
5. Current trends and debates in this area
6. Potential research gaps

Format your response as a structured analysis."""

        response = self.client.generate(prompt, self.system_prompt)
        return {"analysis": response.content, "model": response.model, "usage": response.usage}

    def generate_research_questions(
        self, topic: str, research_type: str, context: str = ""
    ) -> List[str]:
        """Generate research questions using LLM"""
        prompt = f"""Generate focused research questions for the following:

Topic: {topic}
Research Type: {research_type}
Additional Context: {context}

Generate 5-7 specific, focused, and answerable research questions appropriate for a {research_type}.
Each question should be:
- Clear and specific
- Researchable with available methods
- Relevant to the topic
- Appropriately scoped

Format: Return each question on a new line, numbered 1-7."""

        response = self.client.generate(prompt, self.system_prompt)

        # Parse questions from response
        lines = response.content.strip().split("\n")
        questions = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering and clean up
                question = line.lstrip("0123456789.-) ").strip()
                if question and "?" in question:
                    questions.append(question)

        return questions[:7]  # Limit to 7 questions

    def recommend_methodology(
        self, topic: str, research_question: str, discipline: str
    ) -> Dict[str, Any]:
        """Get methodology recommendations from LLM"""
        prompt = f"""Recommend the most appropriate research methodology for:

Topic: {topic}
Research Question: {research_question}
Discipline: {discipline}

Provide detailed recommendations including:
1. Primary methodology recommendation with rationale
2. Alternative methodologies to consider
3. Data collection methods
4. Analysis techniques
5. Sample size considerations
6. Potential limitations
7. Ethical considerations

Be specific and justify each recommendation based on the research question and discipline."""

        response = self.client.generate(prompt, self.system_prompt)
        return {"recommendations": response.content, "model": response.model}

    def generate_search_strategy(self, topic: str, discipline: str) -> Dict[str, Any]:
        """Generate literature search strategy using LLM"""
        prompt = f"""Create a comprehensive literature search strategy for:

Topic: {topic}
Discipline: {discipline}

Provide:
1. Recommended databases (with rationale for each)
2. Primary search terms and keywords
3. Boolean search string examples
4. MeSH terms or subject headings (if applicable)
5. Inclusion criteria
6. Exclusion criteria
7. Date range recommendations
8. Grey literature sources to consider

Format as a structured search protocol."""

        response = self.client.generate(prompt, self.system_prompt)
        return {"strategy": response.content, "model": response.model}

    def synthesize_findings(self, papers: List[Dict[str, str]], research_question: str) -> str:
        """Synthesize findings from multiple papers"""
        papers_text = "\n\n".join(
            [
                f"Title: {p.get('title', 'Unknown')}\n"
                f"Authors: {p.get('authors', 'Unknown')}\n"
                f"Abstract: {p.get('abstract', 'No abstract available')}"
                for p in papers[:10]  # Limit to 10 papers to fit context
            ]
        )

        prompt = f"""Synthesize the following research papers in relation to this research question:

Research Question: {research_question}

Papers:
{papers_text}

Provide a synthesis that:
1. Identifies common themes across papers
2. Notes areas of agreement and disagreement
3. Highlights methodological approaches used
4. Identifies gaps in the current literature
5. Suggests implications for future research

Write in an academic style suitable for a literature review section."""

        response = self.client.generate(prompt, self.system_prompt)
        return response.content

    def evaluate_quality(self, methodology: str, research_design: str) -> Dict[str, Any]:
        """Evaluate research quality using LLM"""
        prompt = f"""Evaluate the quality and rigor of this research design:

Methodology: {methodology}
Research Design: {research_design}

Assess:
1. Internal validity
2. External validity
3. Reliability
4. Methodological appropriateness
5. Potential biases
6. Strengths
7. Weaknesses
8. Recommendations for improvement

Provide a quality score (1-10) with detailed justification."""

        response = self.client.generate(prompt, self.system_prompt)
        return {"evaluation": response.content, "model": response.model}

    def close(self):
        """Close the LLM client"""
        self.client.close()


# Recommended models for research tasks
RECOMMENDED_MODELS = {
    "best_quality": {
        "ollama": [
            {
                "name": "llama3.1:70b",
                "vram": "48GB+",
                "context": "128K",
                "notes": "Best quality, needs high-end GPU",
            },
            {
                "name": "qwen2.5:32b",
                "vram": "24GB+",
                "context": "128K",
                "notes": "Excellent for research, Apache-2.0",
            },
            {
                "name": "mixtral:8x7b",
                "vram": "24GB+",
                "context": "32K",
                "notes": "MoE, good throughput",
            },
        ],
        "description": "Best quality models for comprehensive research analysis",
    },
    "balanced": {
        "ollama": [
            {
                "name": "llama3.1:8b",
                "vram": "8GB+",
                "context": "128K",
                "notes": "Great balance of quality/speed",
            },
            {
                "name": "qwen2.5:14b",
                "vram": "12GB+",
                "context": "128K",
                "notes": "Strong reasoning, Apache-2.0",
            },
            {
                "name": "gemma2:9b",
                "vram": "8GB+",
                "context": "8K",
                "notes": "Efficient, good quality",
            },
        ],
        "description": "Balanced models for most research tasks",
    },
    "lightweight": {
        "ollama": [
            {
                "name": "llama3.2:3b",
                "vram": "4GB+",
                "context": "128K",
                "notes": "Small but capable",
            },
            {
                "name": "phi4:3.8b",
                "vram": "4GB+",
                "context": "128K",
                "notes": "Excellent reasoning for size",
            },
            {"name": "qwen2.5:7b", "vram": "6GB+", "context": "128K", "notes": "Good for laptops"},
        ],
        "description": "Lightweight models for laptops and limited hardware",
    },
    "reasoning": {
        "ollama": [
            {
                "name": "deepseek-r1:7b",
                "vram": "8GB+",
                "context": "64K",
                "notes": "Strong reasoning/math",
            },
            {
                "name": "phi4-reasoning:14b",
                "vram": "12GB+",
                "context": "32K",
                "notes": "Chain-of-thought",
            },
            {
                "name": "qwen2.5-coder:7b",
                "vram": "6GB+",
                "context": "128K",
                "notes": "Good for data analysis",
            },
        ],
        "description": "Models optimized for reasoning and analysis tasks",
    },
}


def get_model_recommendations(vram_gb: int = 8) -> Dict[str, List[Dict]]:
    """Get model recommendations based on available VRAM"""
    recommendations = {}

    for category, data in RECOMMENDED_MODELS.items():
        suitable_models = []
        for model in data["ollama"]:
            # Parse VRAM requirement
            vram_req = int(model["vram"].replace("GB+", ""))
            if vram_req <= vram_gb:
                suitable_models.append(model)

        if suitable_models:
            recommendations[category] = {
                "models": suitable_models,
                "description": data["description"],
            }

    return recommendations


def main():
    """Example usage of LLM provider"""
    # Check if Ollama is available
    config = LLMConfig(
        provider=LLMProvider.OLLAMA, model="llama3.1:8b", base_url="http://localhost:11434"
    )

    try:
        client = OllamaClient(config)
        models = client.list_models()
        print(f"Available Ollama models: {models}")

        if models:
            # Test generation
            assistant = ResearchLLMAssistant(config)
            result = assistant.analyze_topic(
                "Impact of remote work on employee productivity", "psychology"
            )
            print("\n=== Topic Analysis ===")
            print(result["analysis"])
            assistant.close()
        else:
            print("No models available. Pull a model first:")
            print("  ollama pull llama3.1:8b")

        client.close()
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("\nMake sure Ollama is running:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Start Ollama: ollama serve")
        print("  3. Pull a model: ollama pull llama3.1:8b")

    # Print model recommendations
    print("\n=== Model Recommendations ===")
    recs = get_model_recommendations(vram_gb=12)
    for category, data in recs.items():
        print(f"\n{category.upper()}: {data['description']}")
        for model in data["models"]:
            print(f"  - {model['name']} ({model['vram']}, {model['context']} context)")


if __name__ == "__main__":
    main()
