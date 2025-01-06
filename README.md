# Document Q&A System with Gemini

A multi-user document search and conversational Q&A system using Google's Gemini model. Used for chatting with call transcript of earning call for zomato, KRN and Eicher motor.

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
4. Place your PDF documents in the `data/` directory. (If you are doing this need to run the preprocess code to create indexes, currently sample data for 3 companies have been put)
5. Configure user access in `docs_config.json` by mapping email addresses to allowed company documents

## Building the Index

Run the preprocessing script to generate FAISS indexes for your PDFs: (currently in the demo, the indexes are created, no need to run this file.)

After putting the gemini key directly - run ( streamlit run app.py )