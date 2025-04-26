from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict
import json
import os

# Load environment variables
load_dotenv()
print("Loaded API Key:", os.getenv("OPENAI_API_KEY"))
client = OpenAI()

# STEP 1: Load & Split PDF
pdf_path = "/Users/dhananjaygupta/Documents/Developer/GenAI-Cohort/GenAI/FinalDraft.pdf"
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

# STEP 3: Multi-query rewriting using GPT-4
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

# STEP 4: Use Reciprocal Rank Fusion for combined retrieval
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

def reciprocal_rank_fusion(results_per_query, k=60):
    scores = defaultdict(float)

    for result_list in results_per_query:
        for rank, doc in enumerate(result_list):
            doc_id = doc.page_content  # using page content as unique ID
            scores[doc_id] += 1 / (k + rank + 1)

    # Sort by score descending
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Map back to Document objects
    doc_lookup = {
        doc.page_content: doc
        for result_list in results_per_query
        for doc in result_list
    }
    return [doc_lookup[doc_id] for doc_id, _ in sorted_docs]

def retrieve_combined_context(queries, k=5):
    results_per_query = []
    for q in queries:
        results = retriever.similarity_search(query=q, k=k)
        results_per_query.append(results)
    
    return reciprocal_rank_fusion(results_per_query)

# STEP 5: Ask question
def ask_question(user_query):
    variants = generate_query_variants(user_query)
    print("Query Variants:", variants)

    chunks = retrieve_combined_context(variants)
    context = "\n\n".join([doc.page_content for doc in chunks[:10]])  # limit to 10 chunks for token safety

    SYSTEM_PROMPT = f"""
    You are a helpful AI Assistant who answers questions using only the provided context.
    If you can't find the answer in the context, say "I don't know based on the context."

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
    print("Answer:\n", answer)
