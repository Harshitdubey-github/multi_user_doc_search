import os
import streamlit as st
import json
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables (Google Gemini API key)
load_dotenv()

# 1. Session state initialization
def initialize_session_state():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = ""
    if "chain" not in st.session_state:
        st.session_state["chain"] = None
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []

# 2. Load FAISS indexes for companies the user can access
def load_faiss_index(embeddings_folder="embeddings"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load user-access config
    user_email = st.session_state.get("user_email", "")
    with open("docs_config.json", "r") as f:
        access_config = json.load(f)
    allowed_companies = access_config.get(user_email, [])

    combined_vectorstore = None
    for company in allowed_companies:
        try:
            company_vectorstore = FAISS.load_local(
                folder_path=embeddings_folder,
                index_name=company,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            if combined_vectorstore is None:
                combined_vectorstore = company_vectorstore
            else:
                combined_vectorstore.merge_from(company_vectorstore)
        except Exception as e:
            st.warning(f"Could not load index for {company}: {str(e)}")
            continue

    if combined_vectorstore is None:
        raise RuntimeError("No valid indexes could be loaded for this user.")

    return combined_vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Initialize the ConversationalRetrievalChain
def initialize_chain():
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY not found in environment variables")
            return None

        retriever = load_faiss_index()
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        return chain
    except Exception as e:
        st.error(f"Error initializing chain: {str(e)}")
        return None

# 4. Handle user queries
def handle_user_query(user_query: str):
    chain = st.session_state["chain"]
    if not chain:
        st.error("Chain is not initialized. Please log in again.")
        return

    try:
        # Pass recent conversation as chat history
        # We remove 'Sources:' from the assistant's prior answers
        recent_history = [
            (q, a.split("\n\nSources:")[0]) 
            for (q, a) in st.session_state["conversation_history"][-3:]
        ]

        result = chain.invoke({
            "question": user_query,
            "chat_history": recent_history
        })
        answer = result.get("answer", "Sorry, I couldn't find an answer.")
        source_docs = result.get("source_documents", [])

        # Format sources (if any)
        if source_docs:
            source_text = "\n\nSources:\n"
            for i, doc in enumerate(source_docs, 1):
                snippet = doc.page_content[:200].replace("\n", " ")
                source_text += f"{i}. {snippet}...\n"
            answer += source_text

        # Save the conversation pair to session
        st.session_state["conversation_history"].append((user_query, answer))
        st.experimental_rerun()  # Refresh for immediate display
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

# 5. Login form
def show_login_form():
    st.subheader("Login Simulation")
    email_input = st.text_input("Enter your email")

    if st.button("Login"):
        with open("docs_config.json", "r") as f:
            access_config = json.load(f)

        if email_input in access_config:
            st.session_state["logged_in"] = True
            st.session_state["user_email"] = email_input

            chain = initialize_chain()
            if chain is not None:
                st.session_state["chain"] = chain
                st.success(f"Logged in as {email_input}.")
                st.experimental_rerun()
            else:
                st.error("Failed to initialize chain. Check your API key.")
                st.session_state["logged_in"] = False
        else:
            st.error("Invalid email or no access configured.")

# 6. The main chat interface
def show_chat_interface():
    # -- Custom CSS for color-coded chat bubbles --
    st.markdown(
        """
        <style>
        .user-message {
            background-color: #007AFF; /* Blue for user */
            color: white;
            padding: 12px 18px;
            border-radius: 15px 15px 2px 15px;
            margin-bottom: 12px;
            max-width: 80%;
        }
        .assistant-message {
            background-color: #f0f0f0; /* Gray for assistant */
            color: #1f1f1f;
            padding: 12px 18px;
            border-radius: 15px 15px 15px 2px;
            margin-bottom: 12px;
            max-width: 80%;
        }
        .info-box {
            background-color: #f8f9fa;
            padding: 10px 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            margin-bottom: 20px;
        }
        .company-badge {
            display: inline-block;
            background-color: #007AFF;
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            margin: 0 4px;
            font-size: 0.9em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Show user info and company access
    with open("docs_config.json", "r") as f:
        access_config = json.load(f)
    user_companies = access_config.get(st.session_state["user_email"], [])

    st.markdown(
        f"""
        <div class="info-box">
            <p style="margin-bottom: 8px;">
                <strong>👤 {st.session_state['user_email']}</strong>
            </p>
            <p style="margin: 0;">
                🔑 Access to: {' '.join(
                    [f'<span class="company-badge">{company}</span>' for company in user_companies]
                )}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display conversation (user on the right, assistant on the left)
    for question, answer in st.session_state["conversation_history"]:
        # User message
        with st.chat_message("user"):
            st.markdown(
                f'<div class="user-message">{question}</div>',
                unsafe_allow_html=True
            )

        # Assistant response
        with st.chat_message("assistant"):
            if "\n\nSources:" in answer:
                main_answer, sources = answer.split("\n\nSources:", 1)
                st.markdown(
                    f'<div class="assistant-message">{main_answer.strip()}</div>',
                    unsafe_allow_html=True
                )
                with st.expander("📚 View Sources"):
                    st.write(sources.strip())
            else:
                st.markdown(
                    f'<div class="assistant-message">{answer}</div>',
                    unsafe_allow_html=True
                )

    # Chat input at the bottom
    user_query = st.chat_input("Type your message here...")
    if user_query:
        handle_user_query(user_query)

    # Clear chat button
    if st.button("Clear Chat", type="secondary"):
        st.session_state["conversation_history"] = []
        st.experimental_rerun()

# 7. Main function
def main():
    st.title("Document-based Chatbot - Harshit")
    initialize_session_state()

    if not st.session_state["logged_in"]:
        show_login_form()
    else:
        show_chat_interface()

if __name__ == "__main__":
    main()
