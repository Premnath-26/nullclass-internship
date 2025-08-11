import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import torch

def load_cs_papers(json_path, chunk_size=10000, max_chunks=10):
    chunks = []
    for i, chunk in enumerate(pd.read_json(json_path, lines=True, chunksize=chunk_size)):
        chunks.append(chunk)
        if i+1 >= max_chunks:
            break
    df = pd.concat(chunks, ignore_index=True)
    df_cs = df[df['categories'].str.contains('cs', na=False)]
    return df_cs.reset_index(drop=True)

def build_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embedder, embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def semantic_search(query, embedder, index, texts, top_k=5):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    results = [texts[i] for i in I[0]]
    return results

def get_summarizer(model_name="facebook/bart-large-cnn"):
    summarizer = pipeline("summarization", model=model_name, device=0 if torch.cuda.is_available() else -1)
    return summarizer

def summarize_text(summarizer, text, max_length=150):
    summary = summarizer(text, max_length=max_length, min_length=40, do_sample=False)
    return summary[0]['summary_text']

def get_explainer(model_name="google/flan-t5-large"):
    explainer = pipeline("text2text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1)
    return explainer

def generate_explanation(explainer, query, contexts, max_length=200):
    context_text = " ".join(contexts)
    input_text = f"Question: {query}\nContext: {context_text}\nAnswer:"
    output = explainer(input_text, max_length=max_length, do_sample=False)
    return output[0]['generated_text']
