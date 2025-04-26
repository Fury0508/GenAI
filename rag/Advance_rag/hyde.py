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

# STEP 3: HyDE — Generate hypothetical answer
def generate_hypothetical_doc(query):
    system_prompt = "Generate a detailed and factual answer to the following question as if you were answering it directly."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content.strip()

# STEP 4: Embed hypo doc and retrieve similar chunks
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

def retrieve_context_hyde(user_query, k=5):
    hypo_doc = generate_hypothetical_doc(user_query)
    print("\n[HyDE] Hypothetical document:\n", hypo_doc)

    hypo_embedding = embedder.embed_query(hypo_doc)

    results = retriever.similarity_search_by_vector(hypo_embedding, k=k)
    return results

# STEP 5: Final answer with retrieved context
def ask_question(user_query):
    chunks = retrieve_context_hyde(user_query)
    context = "\n\n".join([doc.page_content for doc in chunks])

    SYSTEM_PROMPT = f"""
    You are a helpful AI Assistant who responds based only on the provided context.
    If the answer is not in the context, reply with "I don't know based on the context."

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
    answer = ask_question(user_query)
    print("\nAnswer:\n", answer)
