import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()
from sentence_transformers import SentenceTransformer

DATA_DIR = "data/raw"
INDEX_DIR = "embeddings/faiss_index"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class DirectEmbeddings:
    def embed_documents(self, texts):
        return model.encode(texts).tolist()
    def embed_query(self, text):
        return model.encode(text).tolist()

def clean_text(text):
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)

def load_documents():
    docs = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    for file in os.listdir(DATA_DIR):
        if not file.endswith(".txt"):
            continue

        character = file.replace(".txt", "")
        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            text = clean_text(f.read())

        chunks = splitter.split_text(text)

        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "character": character,
                        "source": file
                    }
                )
            )
    return docs

def main():
    embeddings = DirectEmbeddings()

    docs = load_documents()
    print(f"Embedding {len(docs)} chunks...")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_DIR)

    print("FAISS index saved.")

if __name__ == "__main__":
    main()
