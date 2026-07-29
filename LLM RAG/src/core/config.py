import os

# Force Hugging Face to use locally cached models — prevents connection timeout errors
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=os.path.join(_base_dir, ".env"), env_file_encoding="utf-8", extra="ignore")

    # API Keys
    gemini_api_key: str = "your_gemini_api_key_here"
    
    # Neo4j Settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    
    # Models
    llm_model: str = "gemini-3.1-flash-lite"
    embeddings_model: str = "all-MiniLM-L6-v2"
    
    # Feature Flags
    use_dynamic_thresholds: bool = True
    use_bm25: bool = True
    use_graph_retrieval: bool = True
    use_fact_checker: bool = True
    use_intent_classifier: bool = True
    
    # Retrieval Limits
    graph_retrieval_limit: int = 5
    vector_retrieval_limit: int = 5
    
    @property
    def base_dir(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
    @property
    def data_dir(self) -> str:
        """Points to the local data folder."""
        return os.path.join(self.base_dir, "data")
        
    @property
    def chroma_db_path(self) -> str:
        return os.path.join(self.base_dir, "chroma_db")

    @property
    def bm25_index_path(self) -> str:
        """Path to persist the BM25 sparse index."""
        return os.path.join(self.base_dir, "chroma_db", "bm25_index.pkl")

# Singleton instance
settings = Settings()

def get_llm():
    return ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0.01, api_key=settings.gemini_api_key)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=settings.embeddings_model)
