from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI
import json
load_dotenv()
client = OpenAI()
pdf_path = Path(__file__).parent / "FinalDraft.pdf"

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

split_docs = text_splitter.split_documents(documents=docs)
embedder = OpenAIEmbeddings(
    model = "text-embedding-3-large",
    # api_key="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"
)

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url = "http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embedder
# )

# vector_store.add_documents(documents = split_docs)
print("Ingestion done")

# print("DOCS",len(docs))
# print("SPLIT",len(split_docs))

retriver = QdrantVectorStore.from_existing_collection(
    url = "http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

relevant_chunks  = retriver.similarity_search(
    query="What is BERT?"
)
# print("relevant chunks ",relevant_chunks)

SYSTEM_PROMPT  = f"""
You are a helpful AI Assistant who respond base of the avaliable context


context:
{relevant_chunks}

"""

result= client.chat.completions.create(
    model="gpt-4",
    messages=[
        { "role":"system", "content":SYSTEM_PROMPT},
        {"role":"user","content":"what is bert?"}
    ]
)

print(result.choices[0].message.content)