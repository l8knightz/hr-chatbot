from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DOCS_DIR = Path("docs")
PERSIST_DIR = "chroma_store"

CHUNK = 1200
OVERLAP = 200

def load_docs(folder: Path):
    docs = []
    for pdf in sorted(folder.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf))
        for d in loader.load():
            d.metadata = d.metadata or {}
            d.metadata["source"] = pdf.name
            docs.append(d)
    return docs

def build_store():
    docs = load_docs(DOCS_DIR)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK, chunk_overlap=OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "]
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectordb.persist()
    return vectordb

if __name__ == "__main__":
    Path(PERSIST_DIR).mkdir(exist_ok=True)
    build_store()
    print("Index built ✔")
