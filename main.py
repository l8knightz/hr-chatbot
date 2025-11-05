from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from chromadb.config import Settings
from ingest import build_store, PERSIST_DIR
from prompts import SYSTEM_PROMPT

load_dotenv()

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def get_store():
    persist = Path(PERSIST_DIR)
    if not persist.exists() or not any(persist.iterdir()):
        print("No Chroma store found — building from PDFs in ./docs ...")
        build_store()
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False),
    )

store = get_store()
retriever = store.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)

def answer_question(question: str):
    docs = retriever.invoke(question)
    if not docs:
        return "I couldn't find relevant context in the documents to answer that."
    ctx_lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        preview = d.page_content.strip().replace("\n", " ")[:750]
        ctx_lines.append(f"[S{i}] from {src}:\n{preview}\n")
    context = "\n".join(ctx_lines)

    messages = prompt.format_messages(question=question, context=context)
    resp = llm.invoke(messages)
    return resp.content

with gr.Blocks(title="PDF Chatbot — RAG (Chroma + OpenAI)") as demo:
    gr.Markdown("""
    # 📄 Conversational PDF Chatbot
    Ask questions about the documents in the `docs/` folder. Answers include inline citations like [S1], [S2].
    """)
    chat = gr.ChatInterface(
        fn=lambda msg, hist: answer_question(msg),
        title="Ask about the PDFs",
        textbox=gr.Textbox(placeholder="e.g., What topics does the Nestlé HR policy cover?"),
        theme="soft",
        type="messages",
    )

if __name__ == "__main__":
    demo.launch()
