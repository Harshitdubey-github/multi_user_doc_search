Setup:

Clone the repo.
pip install -r requirements.txt.
Place or confirm sample PDFs in data/.
Update docs_config.json with your own user-to-company mappings if desired.
Build Index:

python preprocess.py to generate FAISS indexes for the sample PDFs.
Run App:

streamlit run app.py
Usage:

Enter a valid email from docs_config.json.
Ask questions and see the relevant excerpts.
Notes:

This is a simplified demonstration. Real-world usage would incorporate robust user authentication, database connections, and possibly an LLM to refine answers.