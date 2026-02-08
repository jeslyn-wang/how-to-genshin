import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Configuration
INDEX_DIR = "embeddings/faiss_index"
# Ensure your .env has HUGGINGFACE_API_TOKEN
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# 1. Setup Embeddings (Must match ingest.py)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load Vector Store
if os.path.exists(INDEX_DIR):
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
else:
    print(f"Index not found at {INDEX_DIR}. Please run ingest.py first.")
    exit()

# 3. Setup LLM with Chat Wrapper
llm_endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.1,
)
llm = ChatHuggingFace(llm=llm_endpoint)

def ask(question):
    # Retrieve context
    docs = retriever.invoke(question)
    
    context = "\n\n".join(
        f"[{d.metadata.get('character', 'Unknown')}]\n{d.page_content}"
        for d in docs
    )

    # ChatHuggingFace handles the Llama-3 formatting for you
    messages = [
        SystemMessage(content=(
            "You are a Genshin Impact build assistant. "
            "Use ONLY the information in the context below. "
            "If the answer is not present, say you don't know."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}")
    ]

    response = llm.invoke(messages)
    return response.content

def main():
    print("--- Genshin Build Assistant Ready ---")
    print("(Type 'exit' to quit)")
    
    while True:
        user_query = input("\nUser: ")
        
        if user_query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
            
        if not user_query.strip():
            continue

        try:
            answer = ask(user_query)
            print(f"\nAssistant: {answer}")
        except Exception as e:
            print(f"\nError: {e}")
            print("Tip: Check your internet or ensure you have access to the Llama-3 model on Hugging Face.")

if __name__ == "__main__":
    main()