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

# A small helper to load FAISS index
def load_faiss_index(embeddings_folder="embeddings"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Debug information
    print("Available files in embeddings folder:")
    print(os.listdir(embeddings_folder))
    
    # Create a combined vectorstore
    combined_vectorstore = None
    
    # Get the user's allowed companies
    user_email = st.session_state.get("user_email", "")
    with open("docs_config.json", "r") as f:
        access_config = json.load(f)
    allowed_companies = access_config.get(user_email, [])
    print("Allowed companies:", allowed_companies)
    
    # Load and merge indexes for allowed companies
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
        raise RuntimeError("No valid indexes could be loaded")
        
    return combined_vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize LangChain chain
def initialize_chain():
    try:
        # Debug information
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY not found in environment variables")
            return None

        retriever = load_faiss_index()
        
        # Updated to use Gemini model with error handling
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Changed from gemini-1.5-pro to gemini-pro
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

def initialize_session_state():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = ""
    if "chain" not in st.session_state:
        st.session_state["chain"] = None
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []

def main():
    st.title("Multi-User Document Search & Conversational Q&A")

    initialize_session_state()

    # Load user-access config
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
            try:
                # Set session state
                st.session_state["logged_in"] = True
                st.session_state["user_email"] = email_input

                # Initialize LangChain chain with error handling
                chain = initialize_chain()
                if chain is not None:
                    st.session_state["chain"] = chain
                    st.success(f"Logged in as {email_input}. Access configured.")
                    st.rerun()  # Force a rerun to refresh the page
                else:
                    st.error("Failed to initialize chain. Please check your API key and try again.")
                    st.session_state["logged_in"] = False
                    
            except Exception as e:
                st.error(f"Error during login: {str(e)}")
                st.session_state["logged_in"] = False
        else:
            st.error("Invalid email or no access configured.")

def show_search_interface():
    st.subheader("Conversational Q&A")
    
    # Check if chain is properly initialized
    if st.session_state.get("chain") is None:
        st.error("Chain is not initialized. Please log out and log in again.")
        if st.button("Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
        return
        
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
    chain = st.session_state["chain"]
    if not chain:
        st.error("Chain is not initialized. Please log in again.")
        return
    
    # Run the query through LangChain
    result = chain({"question": user_query, "chat_history": st.session_state["conversation_history"]})
    answer = result["answer"]
    
    # Save to conversation history
    st.session_state["conversation_history"].append((user_query, answer))
    st.success(answer)

if __name__ == "__main__":
    main()
