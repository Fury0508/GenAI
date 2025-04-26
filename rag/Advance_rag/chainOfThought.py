from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables
load_dotenv()
print("Loaded API Key:", os.getenv("OPENAI_API_KEY"))
client = OpenAI()

# STEP 1: Load & Split PDF
pdf_path = Path(__file__).parent / "FinalDraft.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(documents=docs)

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv('OPENAI_API_KEY')
)

# STEP 2: Ingest into Qdrant (run once)
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)
print("Ingestion done")

# STEP 3: Create retriever from existing collection
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

# STEP 4: Use direct query to retrieve context
def retrieve_context_direct(user_query, k=5):
    return retriever.similarity_search(query=user_query, k=k)

# STEP 5: Ask question with Chain of Thought (CoT) prompting
def ask_question(user_query, abstract_level="less"):
    chunks = retrieve_context_direct(user_query)
    context = "\n\n".join([doc.page_content for doc in chunks])

    # Chain of Thought prompt style
    if abstract_level == "less":
        reasoning_instruction = "Think step-by-step in a clear, detailed, and concrete manner to arrive at the answer."
    else:
        reasoning_instruction = "Think in a high-level, abstract, and conceptual manner to derive your answer."

    SYSTEM_PROMPT = f"""
    You are a helpful AI assistant who responds only based on the provided context.
    If the answer is not in the context, reply with "I don't know based on the context."

    {reasoning_instruction}

    context:
    {context}
    """

    result = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
    )

    return result.choices[0].message.content

# ✅ Example usage
if __name__ == "__main__":
    user_query = "What is BERT?"

    print("\n--- Less Abstract (Concrete CoT) ---")
    answer_concrete = ask_question(user_query, abstract_level="less")
    print(answer_concrete)

    print("\n--- More Abstract (Conceptual CoT) ---")
    answer_abstract = ask_question(user_query, abstract_level="more")
    print(answer_abstract)
