from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import torch
import pandas as pd
import numpy as np
import pickle
import os
import PyPDF2
import re
import faiss
from sklearn.decomposition import PCA
from langchain_community.document_loaders import PyPDFLoader
from process_research_paper import PDFCleaner, process_pdf
import shutil

def build_combined_store(
    clinical_data_dir: str = "data/clinical_embeddings",
    papers_dir: str = "data/research_papers",
    vector_store_dir: str = "data/vector_store",
    research_weight: float = 0.5
):
    """
    Build a combined vector store using both HMS clinical embeddings and research papers.
    Uses PDFCleaner for cleaning research paper text.
    
    Args:
        clinical_data_dir: Directory containing HMS embeddings and metadata
        papers_dir: Directory containing research papers
        vector_store_dir: Directory to save the combined vector store
        research_weight: Weight for research vs clinical embeddings
    """
    print("Building combined vector store...")
    
    # Clear existing vector store directory if it exists
    if os.path.exists(vector_store_dir):
        print(f"\nClearing existing vector store at {vector_store_dir}...")
        shutil.rmtree(vector_store_dir)
    
    # Initialize PDF cleaner
    pdf_cleaner = PDFCleaner()
    
    # Load HMS embeddings and metadata
    with open(os.path.join(clinical_data_dir, "full_h_embed_hms.pkl"), "rb") as f:
        hms_embeddings = pickle.load(f)
    print("Successfully loaded HMS embeddings from clinical_chatbot/data/clinical_embeddings/full_h_embed_hms.pkl")
    print(f"HMS embeddings shape: {hms_embeddings.shape}")
    
    metadata_df = pd.read_csv(os.path.join(clinical_data_dir, "new_node_map_df.csv"))
    print("Successfully loaded metadata from clinical_chatbot/data/clinical_embeddings/new_node_map_df.csv")
    print(f"Number of nodes: {len(metadata_df)}")
    
    print("\nLoading clinical data...")
    clinical_embeddings = hms_embeddings
    print("Successfully loaded clinical data:")
    print(f"- Embeddings shape: {clinical_embeddings.shape}")
    print(f"- Number of nodes: {len(metadata_df)}")
    
    print("\nConverting clinical data to documents...")
    clinical_documents = []
    for idx, row in metadata_df.iterrows():
        doc = Document(
            page_content=row['node_name'],
            metadata={
                'source': 'clinical',
                'node_id': row['node_id'],
                'ntype': row['ntype'],
                'global_index': row['global_graph_index'],
                'embedding': clinical_embeddings[idx]
            }
        )
        clinical_documents.append(doc)
    print(f"Created {len(clinical_documents)} clinical documents")
    
    print("\nProcessing research papers...")
    research_documents = []
    research_texts = []
    
    # Process each research paper
    for paper_path in ["data/research_papers/Slamon-DJ-text2.pdf"]:
        print(f"\nProcessing {os.path.basename(paper_path)}...")
        
        # Use process_pdf from research_embeddings.py
        chunks = process_pdf(paper_path, chunk_size=1500, chunk_overlap=100)
        research_documents.extend(chunks)
        research_texts.extend([doc.page_content for doc in chunks])
    
    print(f"\nTotal research documents: {len(research_documents)}")
    
    print("\nGenerating research embeddings...")
    model = HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    
    # Generate embeddings in batches
    batch_size = 32
    research_embeddings = []
    
    for i in range(0, len(research_texts), batch_size):
        batch = research_texts[i:i+batch_size]
        batch_embeddings = model.embed_documents(batch)
        research_embeddings.extend(batch_embeddings)
        print(f"Processed {min(i+batch_size, len(research_texts))}/{len(research_texts)} documents")
    
    # Convert embeddings to numpy arrays
    clinical_embeddings = np.array(clinical_embeddings)
    research_embeddings = np.array(research_embeddings)
    
    print("\nFinding matching clinical terms for each research chunk...")
    K = 5  # Keep top-K matches per chunk
    clinical_dim = K * 128  # Each clinical match is 128-dim, we'll keep K of them
    
    for i, (doc, research_emb) in enumerate(zip(research_documents, research_embeddings)):
        # Find matching clinical terms in this chunk
        chunk_text = doc.page_content.lower()
        matches = []
        
        for idx, row in metadata_df.iterrows():
            term = row['node_name'].lower()
            if term in chunk_text:
                # Score based on term length and frequency
                term_freq = chunk_text.count(term)
                term_length = len(term)
                score = term_length * term_freq
                matches.append((idx, term, score))
        
        # Sort matches by score and take top K
        top_matches = sorted(matches, key=lambda x: x[2], reverse=True)[:K]
        
        # Initialize fixed-size array for clinical embeddings
        matching_clinical_emb = np.zeros((K, 128))
        matching_terms = []
        
        # Fill in the embeddings and terms for top matches
        for j, (idx, term, _) in enumerate(top_matches):
            matching_clinical_emb[j] = clinical_embeddings[idx]
            matching_terms.append(term)
        
        # Combine embeddings: research_emb (768-dim) + K clinical embeddings (K*128-dim)
        doc.metadata['source'] = 'research'
        doc.metadata['embedding'] = np.concatenate([research_emb, matching_clinical_emb.flatten()])
        doc.metadata['matching_terms'] = matching_terms
        print(f"\rProcessed chunk {i+1}/{len(research_documents)} - Found {len(matches)} matches, kept top {len(top_matches)}", end="")
    print("\n")
    
    # Update clinical documents with padded embeddings
    print("Updating clinical document embeddings...")
    for doc in clinical_documents:
        # For clinical documents: pad research part with zeros (768-dim) + current embedding padded to K*128
        clinical_emb = doc.metadata['embedding']
        padded_clinical = np.zeros((K, 128))
        padded_clinical[0] = clinical_emb  # Put the original embedding in first position
        doc.metadata['embedding'] = np.concatenate([np.zeros(768), padded_clinical.flatten()])
    
    # Combine documents
    all_documents = clinical_documents + research_documents
    print(f"\nTotal combined documents: {len(all_documents)}")
    
    print("\nCreating vector store...")
    # Extract embeddings from document metadata
    embeddings_list = [doc.metadata['embedding'] for doc in all_documents]
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]  # Should be 768 + (K*128)
    print(f"Creating FAISS index with dimension {dimension} (768 research + {K}*128 clinical)")
    
    # Create basic FlatL2 index
    index = faiss.IndexFlatL2(dimension)
    
    # Normalize embeddings before adding to index
    embeddings_array = embeddings_array.astype(np.float32)
    norms = np.linalg.norm(embeddings_array, axis=1)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_array = embeddings_array / norms[:, np.newaxis]

    # Add vectors to index
    print("Adding vectors to index...")
    batch_size = 1000
    for i in range(0, len(embeddings_array), batch_size):
        batch = embeddings_array[i:i+batch_size]
        index.add(batch)
        print(f"Added batch {i//batch_size + 1}/{(len(embeddings_array) + batch_size - 1)//batch_size}")
    
    # Save index and documents
    os.makedirs(vector_store_dir, exist_ok=True)
    
    faiss.write_index(index, os.path.join(vector_store_dir, "index.faiss"))
    with open(os.path.join(vector_store_dir, "documents.pkl"), "wb") as f:
        pickle.dump(all_documents, f)
    
    print("\nSuccessfully created and saved vector store!")

if __name__ == "__main__":
    build_combined_store() 