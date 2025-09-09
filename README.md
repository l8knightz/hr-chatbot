# HR Policy PDF Chatbot (RAG + Gradio)

Conversational chatbot that answers questions grounded in PDF documents using:
- **Ingestion**: PyPDFLoader + RecursiveCharacterTextSplitter
- **Vector DB**: Chroma + OpenAI Embeddings
- **LLM**: GPT-3.5-Turbo via LangChain’s ChatOpenAI
- **UI**: Gradio ChatInterface

> This repo ships a Dockerized, no-notebook workflow. It’s easy to run locally and safe to fork.

## ⚠️ Documents & Copyright
This project is designed to work with **your own PDFs**. You may not have redistribution rights for third-party documents.  
Create a `docs/` folder and put your PDFs there locally (they are **not** committed to the repo).

hr-chatbot/
Dockerfile
requirements.txt
main.py
ingest.py
prompts.py
.env.example
README.md
docs/ # <-- put PDFs here locally (not in Git)


## Option A: Run with Docker (recommended)
```bash
docker build -t hr-chatbot .
docker run --rm -it \
  -p 7860:7860 \
  --env-file .env \
  -v "$PWD/docs:/app/docs" \
  hr-chatbot
  ```

## Option B: Run locally (no Docker)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-xxxx                        # or use .env
python main.py
```

## How it works

1. Ingest: PDF → pages → overlapping chunks.
2. Embed: chunks → OpenAI embeddings → Chroma persistent store.
3. Retrieve: similarity search (k=4).
4. Generate: GPT-3.5-Turbo answers only from retrieved context, with inline citations [S1], [S2].
5. UI: Gradio ChatInterface.

## Troubleshooting

* Ensure docs/ contains PDFs locally.
* If you see deprecation warnings from LangChain/Chroma, the app still works; see comments in code for the modern imports (langchain-chroma) and .invoke() usage.