# Arxiv CS Paper Search — Streamlit Semantic Search App

This project is an interactive web application built with **Streamlit** that allows users to perform semantic search over a large dataset of computer science papers from the Arxiv repository.  

It uses **FAISS**, a high-performance similarity search library, to find the most relevant papers based on query embeddings. Additionally, it provides automated **summaries** and **expert explanations** for the top search results using NLP models.

---

## Key Features

- **Load and process Arxiv dataset**:  
  The app uses paper titles and abstracts from Arxiv CS metadata.

- **Embedding generation**:  
  Generates vector embeddings for all papers using a pretrained sentence transformer model for semantic understanding.

- **FAISS index**:  
  Builds a fast approximate nearest neighbor index for efficient search over thousands of paper embeddings.

- **Semantic search**:  
  Finds top-k papers most similar to the user query in semantic space.

- **Summarization**:  
  Uses NLP summarization models to provide concise summaries of retrieved papers.

- **Expert explanations**:  
  Generates natural language explanations that clarify or expand on the query with respect to search results.

- **Interactive Streamlit UI**:  
  User-friendly interface to input queries, view results, summaries, and explanations.

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/arxiv-cs-search.git
cd arxiv-cs-search
