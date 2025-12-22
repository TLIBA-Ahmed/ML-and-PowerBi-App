#!pip install transformers sentence-transformers faiss-cpu langchain langchain-text-splitters streamlit

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

df = pd.read_csv("ai_job_dataset.csv")

documents = []
metadatas = []
ids = []

for i, row in df.iterrows():
    doc_text = (
        f"Job {row['job_id']}: {row['job_title']} at {row['company_name']}. "
        f"Location: {row['company_location']}, employee residence: {row['employee_residence']}. "
        f"Salary: {row['salary_usd']} {row['salary_currency']}. "
        f"Experience level: {row['experience_level']}, years of experience required: {row['years_experience']}. "
        f"Employment type: {row['employment_type']}, company size: {row['company_size']}. "
        f"Remote ratio: {row['remote_ratio']}%. "
        f"Industry: {row['industry']}. "
        f"Required skills: {row['required_skills']}. "
        f"Education required: {row['education_required']}. "
        f"Benefits score: {row['benefits_score']}. "
        f"Job description length: {row['job_description_length']} characters. "
        f"Posting date: {row['posting_date']}, application deadline: {row['application_deadline']}."
    )

    documents.append(doc_text)
    metadatas.append({
        "job_id": row["job_id"],
        "job_title": row["job_title"],
        "company_location": row["company_location"],
        "salary_usd": row["salary_usd"],
        "salary_currency": row["salary_currency"],
        "company_name": row["company_name"],
        "industry": row["industry"]
    })
    ids.append(str(i))

print(f"\nNombre de documents créés : {len(documents)}")

"""Chunking"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.create_documents(documents, metadatas=metadatas)
print(f"Nombre de chunks créés : {len(chunks)}")
print(chunks[0])

"""Embeddings"""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_texts = [chunk.page_content for chunk in chunks]
chunk_embeddings = model.encode(chunk_texts)
print(f"shape of our embeddings : {chunk_embeddings.shape}")

"""Vector store with FAISS"""

import faiss
import numpy as np
d = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(chunk_embeddings).astype('float32'))
print(f"FAISS index created with {index.ntotal} vectors")

"""Retrieve, Augment, Generate"""

import google.generativeai as genai
from google.colab import userdata

# Configure Gemini API key
genai.configure(api_key=userdata.get('GEMINI_API_KEY'))

# Initialize Gemini model
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

def answer_question(query: str) -> str:
  query_embedding = model.encode([query]).astype('float32')
  k=5
  distances, indices = index.search(query_embedding, k)
  retrieved_chunks = [chunks[i].page_content for i in indices[0]]
  context = "\n\n".join(retrieved_chunks)

  prompt_template = f"""
  Answer the following question using *only* the provided context.
  If the answer is not in the context, say "I don't have that information."

  Context:
  {context}
  Question:
  {query}
  Answer:
  """
  print(f"-- Context :\n {context} \n")
  # Use Gemini to generate the answer
  response = gemini_model.generate_content(prompt_template)
  return response.text.strip()

new_question = "Which company offers the highest salary for an AI Engineer?"
answer, context = answer_question(new_question)
print(f"\n\n Answer: {answer}\n\n")
print(f"Context: {context}")

question = "What are the key skills required for data scientist roles ?"
answer, context = answer_question(question)
print(f"\n\n Answer: {answer}\n\n")
print(f"Context: {context}")

answer, context = answer_question('what are the meadian salary of ai engineer')
print(f"\n\n Answer: {answer}\n\n")
print(f"Context: {context}")