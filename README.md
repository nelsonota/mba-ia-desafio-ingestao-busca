# Ingestão e Busca Semântica com LangChain + Postgres

Solução em Python que ingere o conteúdo do `document.pdf` em um banco PostgreSQL com extensão `pgvector` e expõe um chat em linha de comando capaz de responder apenas com base no PDF.

## Requisitos
- Python 3.11+
- Docker + Docker Compose
- Conta e API Key no provedor escolhido (`OpenAI` ou `Google`)

## Setup do ambiente
1. **Instale as dependências locais**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   # source venv/bin/activate # Linux/macOS
   pip install -r requirements.txt
   ```
2. **Configure o `.env`**
   ```bash
   copy .env.example .env    # Windows
   # cp .env.example .env    # Linux/macOS
   ```
   Preencha os campos abaixo (valores padrão entre colchetes):

   | Variável | Descrição |
   |----------|-----------|
   | `DATABASE_URL` | Ex.: `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
   | `PG_VECTOR_COLLECTION_NAME` | Nome da coleção/tabela vetorial (ex.: `mba_ingest`) |
   | `PDF_PATH` | Caminho para o PDF. Default: `document.pdf` |
   | `MODEL_PROVIDER` | `openai` (default) ou `google` |
   | `OPENAI_API_KEY` | Obrigatório se `MODEL_PROVIDER=openai` |
   | `OPENAI_EMBEDDING_MODEL` | Default `text-embedding-3-small` |
   | `OPENAI_LLM_MODEL` | Default `gpt-5-nano` |
   | `GOOGLE_API_KEY` | Obrigatório se `MODEL_PROVIDER=google` |
   | `GOOGLE_EMBEDDING_MODEL` | Default `models/embedding-001` |
   | `GOOGLE_LLM_MODEL` | Default `gemini-2.5-flash-lite` |

## Execução
1. **Suba o banco**
   ```bash
   docker compose up -d
   ```
   O serviço auxiliar `bootstrap_vector_ext` cria automaticamente a extensão `vector`.

2. **Ingestão do PDF**
   ```bash
   python src/ingest.py
   ```
   O script carrega o PDF, divide em chunks de 1000 caracteres (overlap 150), gera embeddings e salva os vetores na coleção configurada.

3. **Chat via CLI**
   ```bash
   python src/chat.py
   ```
   Prompt interativo:
   ```
   Faça sua pergunta (digite 'sair' para encerrar).

   PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
   RESPOSTA: O faturamento foi de 10 milhões de reais.

   PERGUNTA: Quantos clientes temos em 2024?
   RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
   ```

## Estrutura
```
├── docker-compose.yml        # Postgres + pgvector
├── document.pdf              # Fonte de conhecimento
├── requirements.txt
├── .env.example
├── docs/
│   ├── planning.md
│   └── state.md
└── src/
    ├── chat.py               # CLI de perguntas
    ├── config.py             # Configurações compartilhadas
    ├── ingest.py             # Pipeline de ingestão
    └── search.py             # Busca semântica / RAG
```

## Troubleshooting
- **`DATABASE_URL` inválida**: confirme usuário/senha e se o Postgres está ouvindo na porta 5432.
- **Extensão `vector` ausente**: execute `docker compose up -d` novamente ou rode `CREATE EXTENSION vector;`.
- **Resposta sempre fora do contexto**: garanta que `python src/ingest.py` rodou sem erros e que `PG_VECTOR_COLLECTION_NAME` do chat é o mesmo da ingestão.
