import os
from dotenv import load_dotenv
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urljoin
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import Qdrant
# from qdrant_client import QdrantClient

client = OpenAI()
# loader = WebBaseLoader("https://chaidocs.vercel.app/youtube/getting-started/")
# docs = loader.load()
# print(docs) # will print the whole page with metadata
# print(docs[0].metadata) # This will print the metadata


def extract_all_links(url,base_domain = "chaidocs.vercel.app"):
    response = requests.get(url)
    soup = BeautifulSoup(response.text,"html.parser")

    link = []
    for a_tag in soup.find_all('a',href = True):
        href = a_tag['href']
        full_url = urljoin(url,href)
        if base_domain in full_url:
            link.append(full_url)
    return link


base_url = "https://chaidocs.vercel.app/youtube/chai-aur-html/introduction/"

all_links = extract_all_links(base_url)
print(f"found {len(all_links)} links")

all_documents = []

for link in all_links:
    try:
        loader = WebBaseLoader(link)
        documents = loader.load()
        all_documents.extend(documents)
        print(f"Loaded: {link}")
    except Exception as e:
        print("Error in loading the url")
# print(f"Total documents loaded: {len(all_documents)}")

# print(all_documents[10].metadata.get('source').split('/')[-3])


section_titles = []
for index in range(len(all_documents)):
    title = all_documents[index].metadata.get('source').split('/')[-3]

    if "chai-aur" in title:
        section_titles.append(title)

# print(set(section_titles))


# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 200
# )

# split_docs = text_splitter.split_documents(all_documents)


embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv('OPENAI_API_KEY')
)

# SYSTEM_PROMPT_TO_DECIDE_COLLECTION = f"""
#     You are an helpful agent which will take the the documents chunks and based on your understanding of that content you have categoried the chunked document into following section_titles list of options only, dont categories into any other options outside of the list
#     DONT ADD ANY TEXT OTHER THAN THE SECTION TITLES
    
#     section titles:
#     {set(section_titles)}
# """

# for split_docs_index in range(len(split_docs)):
#     response = client.chat.completions.create(
#         model = 'gpt-4.1-mini',
#         messages=[
#             {"role":"system","content": SYSTEM_PROMPT_TO_DECIDE_COLLECTION},
#             {"role": "user","content": split_docs[split_docs_index].page_content}
#         ]
#     )

#     model_response_on_split_chunk = response.choices[0].message.content

#     vector_store = QdrantVectorStore.from_documents(
#         documents=[],
#         url= "http://localhost:6333",
#         collection_name = model_response_on_split_chunk,
#         embedding=embedder

#     )
#     vector_store.add_documents(documents=[split_docs[split_docs_index]])


# print("Ingestion Done")

# retriever = QdrantVectorStore.from_existing_collection(
#     url="http://localhost:6333",
    
#     embedding=embedder
# )


user_query = "What are the common git commands"

SYSTEM_PROMPT_TO_GET_DATA= f"""
You are an helpful AI Assistant who will take the user input and based on the user
query you will categoies the query into provided section titles.

DO NOT ADD ANY TEXT OTHER THAN THE SECTION TITLES

section titled:
{set(section_titles)}

"""


response = client.chat.completions.create(
    model='gpt-4.1',
    messages = [
        {"role": "system","content": SYSTEM_PROMPT_TO_GET_DATA},
        {"role":"user","content": user_query}
    ]
)

print("\n >User query",user_query)
print("\n Assistant Response",response.choices[0].message.content)


collection_name = response.choices[0].message.content
print(collection_name)

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name=collection_name,
    embedding=embedder
)


# vector_store = Qdrant(
#     client=qdrant_client,
#     collection_name=collection_name,
#     embeddings = embedder # openai embedder
# )

results = retriever.similarity_search(user_query)

for r in results:
    print(f"page_content: {r.page_content}, source: {r.metadata.get('source')}")