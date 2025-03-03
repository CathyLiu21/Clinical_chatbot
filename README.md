# Clinical Research Chatbot

A specialized chatbot for querying and analyzing clinical research papers using advanced language models and semantic search.

## Features

- 🔍 **Semantic Search**: Uses BioBERT embeddings for accurate biomedical text understanding
- 🤖 **Advanced LLM**: Powered by Llama 3.2B for generating accurate, context-aware responses
- 📚 **Source Citations**: Provides references to source documents for transparency
- 💻 **User-Friendly Interface**: Clean, modern web interface for easy interaction
- ⚡ **Fast Retrieval**: FAISS vector store for efficient similarity search
- 🎯 **Domain-Specific**: Specialized for clinical and biomedical research papers

## Project Structure

```
clinical_chatbot/
├── backend/
│   ├── app.py                  # Flask server and API endpoints
│   ├── qa_chain.py            # Core QA logic and model integration
│   ├── build_combined_store.py # Vector store creation
│   ├── test_qa_chain.py       # Tests QA chain functionality
│   ├── test_pdf_cleaning.py   # Tests PDF text cleaning and processing
│   └── test_embeddings.py     # Tests embedding generation and combination
├── frontend/
│   └── index.html             # Web interface
├── notebooks/
│   └── research_qa.ipynb      # Step-by-step workflow with visualizations
├── data/
│   ├── research_papers/       # PDF research papers
│   ├── vector_store/         # FAISS index and document store
│   └── clinical_embeddings/   # Clinical data embeddings

```

## Implementations

### 1. Web Application
The main implementation as a web service with:
- Flask backend for API endpoints
- Modern web interface for easy interaction
- Real-time question answering
- Source citations and references

### 2. Jupyter Notebook
An alternative implementation in `notebooks/research_qa.ipynb` featuring:
- Detailed step-by-step workflow explanations
- Interactive visualizations of:
  - Document Chunkings
  - Search results
- Alternative technical approaches:
  - Larger and more powerful LLM model (Llama3 8b)
  - Different embedding models
  - Alternative vector store configurations
  - Alernative/Simplified strategy to combine embeddings
  - Alternative search strategy 
- Experimental features and analysis tools
- Evaluations

## How It Works

1. **Document Processing**:
   - Research papers are processed and embedded using BioBERT
   - Embeddings are combined with clinical knowledge
   - FAISS index is built for efficient similarity search

2. **Question Answering**:
   - User questions are embedded using the same model
   - Relevant documents are retrieved using semantic search
   - Llama 3.2B generates concise, accurate answers
   - Sources are cited for verification

## Testing Components

1. **test_qa_chain.py**: 
   - Tests the complete QA pipeline
   - Verifies answer generation and source retrieval
   - Checks response format and quality

2. **test_pdf_cleaning.py**:
   - Validates PDF text extraction
   - Tests cleaning of scientific terms and formatting
   - Ensures preservation of important identifiers (e.g., HER-2/neu)

3. **test_embeddings.py**:
   - Tests embedding generation for documents
   - Verifies embedding combination process
   - Checks vector dimensions and quality

## Setup and Usage

1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start the Backend Server** (in one terminal):
   ```bash
   cd backend
   source ../venv/bin/activate
   python app.py
   ```
   Backend will run on `http://localhost:5001`

3. **Access the Frontend** (choose one method):
   
   Method A - Direct File:
   - Simply open `frontend/index.html` in your web browser
   
   Method B - HTTP Server (in another terminal):
   ```bash
   cd frontend
   python -m http.server 8000
   ```
   Then visit `http://localhost:8000` in your browser

4. **Using the Chatbot**:
   - The frontend will automatically connect to the backend at `http://localhost:5001`
   - Type your question about the research papers
   - Get answers with relevant source citations
   - Example questions:
     - "What is the role of HER-2/neu in breast cancer?"
     - "How does HER-2/neu amplification affect survival rates?"
     - "What are the clinical implications of HER2 status?"

## Requirements

- Python 3.9+
- PyTorch
- Transformers
- FAISS
- Flask
- Modern web browser

## Note

This chatbot is specifically trained on HER-2/neu breast cancer research papers. The same architecture can be adapted for other research domains by updating the document collection and retraining the vector store. 


