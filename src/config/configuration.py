import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Configuration:
    # LLM
    llm_model: str = field(default_factory=lambda: Configuration.resolve_bedrock_model_id())
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    aws_access_key_id: str = field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", ""))
    aws_secret_access_key: str = field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", ""))
    aws_session_token: str = field(default_factory=lambda: os.getenv("AWS_SESSION_TOKEN", ""))

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
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "./research_history.db"))

    @classmethod
    def from_env(cls) -> "Configuration":
        return cls()

    @staticmethod
    def resolve_bedrock_model_id() -> str:
        """Resolve Bedrock model id from env, stripping quotes/spaces."""
        candidates = [
            os.getenv("BEDROCK_MODEL_ID", ""),
            os.getenv("BEDROCK_LLM_MODEL", ""),
        ]
        for value in candidates:
            cleaned = (value or "").strip().strip('"').strip("'")
            if cleaned:
                return cleaned
        return "qwen.qwen3-32b-v1:0"

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
        """Return a LangChain ChatModel configured for AWS Bedrock."""
        from langchain_aws import ChatBedrockConverse

        model = ChatBedrockConverse(
            model=self.llm_model,
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key_id or None,
            aws_secret_access_key=self.aws_secret_access_key or None,
            aws_session_token=self.aws_session_token or None,
        )
        if structured_output:
            model = model.with_structured_output(structured_output)
        return model
