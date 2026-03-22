import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Configuration:
    # LLM
    llm_model: str = "llama3.1-8b"
    cerebras_api_key: str = field(default_factory=lambda: os.getenv("CEREBRAS_API_KEY", ""))

    # Search
    search_api: str = "tavily"
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    max_search_results: int = 5

    # Supervisor
    max_concurrent_researchers: int = 5
    max_supervisor_iterations: int = 6
    allow_clarification: bool = True

    # Researcher
    max_researcher_tool_calls: int = 10

    # LLM output
    max_structured_output_retries: int = 3
    max_report_tokens: int = 4000

    # Memory
    mem0_api_key: str = field(default_factory=lambda: os.getenv("MEM0_API_KEY", ""))
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "./research_history.db"))

    @classmethod
    def from_env(cls) -> "Configuration":
        return cls()

    @classmethod
    def from_config(cls, config=None) -> "Configuration":
        """Extract Configuration from LangGraph's RunnableConfig."""
        if config and isinstance(config, dict):
            configurable = config.get("configurable", {})
            if isinstance(configurable, cls):
                return configurable
            if isinstance(configurable, dict):
                return cls(**{
                    k: v for k, v in configurable.items()
                    if k in cls.__dataclass_fields__
                })
        return cls()

    def get_model(self, structured_output=None):
        """Return a langchain ChatModel configured for Cerebras."""
        from langchain.chat_models import init_chat_model
        model = init_chat_model(
            model=self.llm_model,
            model_provider="openai",
            base_url="https://api.cerebras.ai/v1",
            api_key=self.cerebras_api_key,
        )
        if structured_output:
            model = model.with_structured_output(structured_output)
        return model
