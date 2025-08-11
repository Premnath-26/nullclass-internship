import streamlit as st
import faiss
from utils import (
    load_cs_papers,
    build_embeddings,
    semantic_search,
    get_summarizer,
    summarize_text,
    get_explainer,
    generate_explanation
)

@st.cache_data
def load_resources():
    df = load_cs_papers("data/arxiv-metadata-oai-snapshot.json")
    texts = (df['title'] + ". " + df['abstract']).tolist()

    embedder, _ = build_embeddings([])  # load embedder only, no new embeddings
    index = faiss.read_index("models/faiss_index.bin")

    summarizer = get_summarizer()
    explainer = get_explainer()
    return df, texts, embedder, index, summarizer, explainer

def main():
    st.title("🧠 CS Research Expert Chatbot")

    df, texts, embedder, index, summarizer, explainer = load_resources()

    query = st.text_input("Enter your research question or topic:")

    if query:
        results = semantic_search(query, embedder, index, texts, top_k=5)

        st.subheader("Relevant Papers:")
        for i, paper in enumerate(results):
            st.markdown(f"**Paper {i+1}:** {paper[:500]}...")
            if st.button(f"Summarize Paper {i+1}"):
                summary = summarize_text(summarizer, paper)
                st.write(f"**Summary:** {summary}")

        if st.button("Get Expert Explanation"):
            explanation = generate_explanation(explainer, query, results)
            st.subheader("Expert Explanation")
            st.write(explanation)

if __name__ == "__main__":
    main()
