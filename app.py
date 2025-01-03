import streamlit as st
import json
import os
import faiss
from utils.embedding_utils import EmbeddingIndex
from utils.pdf_utils import load_company_documents

# A small helper to load indexes once
def load_company_index(company_name, embeddings_folder="embeddings"):
    index_path = os.path.join(embeddings_folder, f"{company_name}.index")
    texts_path = os.path.join(embeddings_folder, f"{company_name}_texts.txt")

    if not os.path.exists(index_path) or not os.path.exists(texts_path):
        st.error(f"Index for {company_name} not found. Please run preprocessing.")
        return None

    # Load the FAISS index
    faiss_index = faiss.read_index(index_path)

    # Load texts
    with open(texts_path, "r", encoding="utf-8") as f:
        doc_texts = [line.strip() for line in f.readlines()]

    # Create an EmbeddingIndex object and attach the loaded FAISS index + doc_texts
    embedding_index = EmbeddingIndex()
    embedding_index.index = faiss_index
    embedding_index.doc_texts = doc_texts

    return embedding_index

def initialize_session_state():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = ""
    if "accessible_indices" not in st.session_state:
        st.session_state["accessible_indices"] = {}
    if "conversation_history" not in st.session_state:
        # We'll store a list of (user_query, system_answer)
        st.session_state["conversation_history"] = []

def main():
    st.title("Multi-User Document Search & Conversational Q&A")

    initialize_session_state()

    # Loading user-access config
    with open("docs_config.json", "r") as f:
        access_config = json.load(f)

    if not st.session_state["logged_in"]:
        login_form(access_config)
    else:
        st.write(f"Logged in as: **{st.session_state['user_email']}**")
        show_search_interface()

def login_form(access_config):
    st.subheader("Login / Access Simulation")
    email_input = st.text_input("Enter your email")
    if st.button("Login"):
        if email_input in access_config:
            # Set session state
            st.session_state["logged_in"] = True
            st.session_state["user_email"] = email_input

            # Based on the user's access, load the relevant indices
            companies_allowed = access_config[email_input]
            st.session_state["accessible_indices"] = {}
            for comp in companies_allowed:
                embedding_index = load_company_index(comp)
                if embedding_index:
                    st.session_state["accessible_indices"][comp] = embedding_index
            
            st.success(f"Logged in as {email_input}. Access to: {companies_allowed}")
        else:
            st.error("Invalid email or no access configured.")

def show_search_interface():
    st.subheader("Conversational Q&A")
    user_query = st.text_input("Your question:")
    if st.button("Ask"):
        if user_query.strip() == "":
            st.warning("Please enter a question.")
        else:
            handle_user_query(user_query)

    # Display conversation history
    if st.session_state["conversation_history"]:
        st.write("### Conversation History")
        for i, (q, ans) in enumerate(st.session_state["conversation_history"]):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {ans}")
            st.write("---")

def handle_user_query(user_query):
    # For a multi-document search, we combine results from all accessible indices
    all_excerpts = []
    for comp, emb_index in st.session_state["accessible_indices"].items():
        results = emb_index.search(user_query, top_k=2)  # top_k is adjustable
        all_excerpts.extend(results)

    # For a “true” conversational answer, you’d pass the entire conversation history +
    # the retrieved excerpts into an LLM. Here, we’ll just show the top excerpt or
    # combine them naively.
    if not all_excerpts:
        answer = "No relevant information found in authorized documents."
    else:
        # For simplicity, we just pick the first excerpt as the "best" answer.
        # In a real system, you'd refine or rank them, or feed into an LLM to generate a summary.
        answer = all_excerpts[0]
    
    # Save to conversation history
    st.session_state["conversation_history"].append((user_query, answer))
    st.success(answer)

if __name__ == "__main__":
    main()
