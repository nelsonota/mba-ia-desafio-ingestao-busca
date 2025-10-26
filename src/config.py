import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

Provider = Literal["openai", "google"]


@dataclass
class Settings:
    database_url: str
    collection_name: str
    provider: Provider
    pdf_path: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 10
    openai_embedding_model: str = "text-embedding-3-small"
    google_embedding_model: str = "models/embedding-001"
    openai_llm_model: str = "gpt-5-nano"
    google_llm_model: str = "gemini-2.5-flash-lite"


def _get_provider() -> Provider:
    provider = os.getenv("MODEL_PROVIDER") or os.getenv("EMBEDDING_PROVIDER") or "openai"
    provider = provider.strip().lower()
    if provider not in {"openai", "google"}:
        raise ValueError("MODEL_PROVIDER deve ser 'openai' ou 'google'.")
    return provider  # type: ignore[return-value]


def _resolve_pdf_path() -> Optional[str]:
    pdf_path = os.getenv("PDF_PATH") or "document.pdf"
    pdf_path = pdf_path.strip()
    return pdf_path or None


def load_settings(*, require_pdf: bool = False) -> Settings:
    database_url = (os.getenv("DATABASE_URL") or "").strip()
    collection_name = (os.getenv("PG_VECTOR_COLLECTION_NAME") or "").strip()
    if not database_url:
        raise ValueError("DATABASE_URL não configurada no .env.")
    if not collection_name:
        raise ValueError("PG_VECTOR_COLLECTION_NAME não configurado no .env.")

    provider = _get_provider()
    pdf_path = _resolve_pdf_path()
    if require_pdf:
        if not pdf_path:
            raise ValueError("PDF_PATH não configurado no .env.")
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF_PATH não encontrado: {pdf_path}")

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY não configurada.")
    if provider == "google" and not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY não configurada.")

    settings = Settings(
        database_url=database_url,
        collection_name=collection_name,
        provider=provider,
        pdf_path=pdf_path,
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        google_embedding_model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"),
        openai_llm_model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano"),
        google_llm_model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite"),
    )
    return settings


def ensure_pdf_path(settings: Settings) -> str:
    if not settings.pdf_path:
        raise ValueError("PDF_PATH não definido. Atualize o .env.")
    pdf = Path(settings.pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf}")
    return str(pdf)


def build_embeddings(settings: Settings):
    if settings.provider == "google":
        return GoogleGenerativeAIEmbeddings(model=settings.google_embedding_model)
    return OpenAIEmbeddings(model=settings.openai_embedding_model)


def build_llm(settings: Settings):
    if settings.provider == "google":
        return ChatGoogleGenerativeAI(model=settings.google_llm_model, temperature=0.0)
    return ChatOpenAI(model=settings.openai_llm_model, temperature=0.0)
