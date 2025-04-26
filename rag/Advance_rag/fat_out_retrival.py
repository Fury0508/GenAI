from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI
import json
import os

load_dotenv()
print("Loaded API Key:", os.getenv("OPENAI_API_KEY"))
client = OpenAI()

# STEP 1: Load & split PDF
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

# STEP 3: Query rewriting using GPT-4
def generate_query_variants(user_query):
    prompt = f"""
    You are a helpful assistant. Generate three different but relevant versions of this query: "{user_query}".
    Return them as a JSON list.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        variants = json.loads(response.choices[0].message.content)
    except:
        variants = [user_query]  # fallback in case JSON parsing fails
    return variants

# STEP 4: Retrieve context for each query
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

def retrieve_combined_context(queries, k=5):
    all_chunks = []
    for q in queries:
        results = retriever.similarity_search(query=q, k=k)
        all_chunks.extend(results)
    # Deduplicate based on page_content
    unique = list({doc.page_content: doc for doc in all_chunks}.values())
    return unique

# STEP 5: Final Answer
def ask_question(user_query):
    variants = generate_query_variants(user_query)
    print("Query Variants:", variants)

    chunks = retrieve_combined_context(variants)
    context = "\n\n".join([doc.page_content for doc in chunks])

    SYSTEM_PROMPT = f"""
    You are a helpful AI Assistant who responds based on the available context.

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
user_query = "What is BERT?"
answer = ask_question(user_query)
print("Answer:\n", answer)
