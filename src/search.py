from typing import Callable, List, Optional, Tuple

from langchain_core.prompts import PromptTemplate
from langchain_postgres import PGVector
from langchain.schema import Document

from config import build_embeddings, build_llm, load_settings

PROMPT_TEMPLATE = """CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

OUT_OF_CONTEXT_MSG = "Não tenho informações necessárias para responder sua pergunta."


def _format_context(
    results: List[Tuple[Document, float]],
) -> Optional[str]:
    if not results:
        return None
    chunks = [doc.page_content.strip() for doc, _ in results if doc.page_content.strip()]
    if not chunks:
        return None
    return "\n\n---\n\n".join(chunks)


def _build_runner() -> Callable[[str], str]:
    settings = load_settings()
    embedding = build_embeddings(settings)
    vector_store = PGVector(
        embeddings=embedding,
        connection=settings.database_url,
        collection_name=settings.collection_name,
    )
    llm = build_llm(settings)
    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=PROMPT_TEMPLATE,
    )

    def runner(question: str) -> str:
        user_question = (question or "").strip()
        if not user_question:
            return "Digite uma pergunta válida."

        results = vector_store.similarity_search_with_score(
            user_question,
            k=settings.top_k,
        )
        contexto = _format_context(results)
        if not contexto:
            return OUT_OF_CONTEXT_MSG

        formatted = prompt.format(contexto=contexto, pergunta=user_question)
        response = llm.invoke(formatted)
        return getattr(response, "content", response) or OUT_OF_CONTEXT_MSG

    return runner


def search_prompt(question: Optional[str] = None):
    """
    Retorna um callable que executa a busca RAG ou, opcionalmente, responde
    imediatamente à pergunta fornecida.
    """
    runner = _build_runner()
    if question is not None:
        return runner(question)
    return runner
