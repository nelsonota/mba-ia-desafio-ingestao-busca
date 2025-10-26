from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import build_embeddings, ensure_pdf_path, load_settings


def ingest_pdf():
    """
    Executa a pipeline de ingestão: carrega o PDF, quebra em chunks, gera embeddings
    e salva vetores no Postgres/pgVector.
    """
    try:
        settings = load_settings(require_pdf=True)
        pdf_path = ensure_pdf_path(settings)
        print(f"📄 Iniciando ingestão do arquivo: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            print("Nenhum conteúdo foi extraído do PDF. Verifique o arquivo.")
            return

        embedding = build_embeddings(settings)
        PGVector.from_documents(
            documents=chunks,
            embedding=embedding,
            connection=settings.database_url,
            collection_name=settings.collection_name,
            pre_delete_collection=True,
        )
        print(
            f"✅ Ingestão concluída ({len(chunks)} chunks) na coleção '{settings.collection_name}'."
        )
    except Exception as exc:
        print(f"❌ Falha durante a ingestão: {exc}")


if __name__ == "__main__":
    ingest_pdf()
