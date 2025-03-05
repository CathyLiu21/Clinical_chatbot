# Clinical Research Chatbot

A specialized chatbot for querying and analyzing clinical research papers using advanced language models and semantic search.

## Features

- 🔍 **Advanced Semantic Search**: 
  - BioBERT embeddings for accurate biomedical text understanding
  - Integration with Harvard's Clinical Knowledge Embeddings (67,124 clinical vocabulary embeddings)
  - Combined embeddings approach for enhanced medical context
- 🤖 **Advanced LLM**: Powered by Llama 3.2B for generating accurate, context-aware responses
- 📚 **Source Citations**: Provides references to source documents for transparency
- 💻 **User-Friendly Interface**: Clean, modern web interface for easy interaction
- ⚡ **Fast Retrieval**: FAISS vector store for efficient similarity search
- 🎯 **Domain-Specific**: 
  - Specialized for clinical and biomedical research papers
  - Enhanced with standardized clinical vocabulary across 7 medical vocabularies
  - No dependency on patient-level information

## Project Structure

```
clinical_chatbot/
├── backend/
│   ├── app.py                  # Flask server and API endpoints
│   ├── qa_chain.py            # Core QA logic and model integration
|   ├── process_research_paper.py  # Preprocess and clean-up and PDF research paper
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

1. **Start the Backend Server** (in one terminal):
   ```bash
   cd backend
   source ../venv/bin/activate
   python app.py
   ```
   Backend will run on `http://localhost:5001`

2. **Access the Frontend** (choose one method):
   
   Method A - Direct File:
   - Simply open `frontend/index.html` in your web browser
   
   Method B - HTTP Server (in another terminal):
   ```bash
   cd frontend
   python -m http.server 8000
   ```
   Then visit `http://localhost:8000` in your browser

3. **Using the Chatbot**:
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

## Citations

1. **Harvard Medical School Clinical Knowledge Embeddings**:
   ```
   @article{beam2020clinical,
     title={Clinical Concept Embeddings Learned from Massive Sources of Medical Data},
     author={Beam, Andrew L and Kompa, Benjamin and Fried, Inbar and Palmer, Nathan Patrick and Shi, Xu and Cai, Tianxi and Kohane, Isaac S},
     journal={arXiv preprint arXiv:2010.00740},
     year={2020}
   }
   ```
   - Source: [Clinical Concept Embeddings (Harvard)](https://arxiv.org/abs/2010.00740)
   - Dataset: [Harvard Medical School - Clinical Embeddings](https://figshare.com/articles/dataset/Clinical_Concept_Embeddings_Learned_from_Massive_Sources_of_Medical_Data/12382343)


