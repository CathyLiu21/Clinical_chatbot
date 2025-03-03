from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import torch
import pickle
import faiss
import os
import numpy as np
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

class QAChain:
    def __init__(
        self,
        vector_store_dir: str = "../data/vector_store",
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        temperature: float = 0.1,
        k: int = 5,  # Number of documents to retrieve
        max_new_tokens: int = 256
    ):
        """Initialize the QA chain with the combined vector store.
        
        Args:
            vector_store_dir: Directory containing the FAISS index and documents
            model_name: HuggingFace model to use (default: llama 3.2 3B Instruct)
            temperature: Temperature for text generation
            k: Number of documents to retrieve
            max_new_tokens: Maximum number of tokens to generate
        """
        self.vector_store_dir = vector_store_dir
        self.k = k
        
        # Load the FAISS index and documents
        print("Loading vector store...")
        self.index = faiss.read_index(os.path.join(vector_store_dir, "index.faiss"))
        with open(os.path.join(vector_store_dir, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        print(f"Loaded {len(self.documents)} documents from vector store")
        
        # Initialize the embedding model (same as in build_combined_store.py)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )
        
        # Convert to GPU if available
        if torch.cuda.is_available():
            print("Using GPU for FAISS search...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Initialize model with optimizations
        print(f"Loading {model_name}...")
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.getenv("HUGGINGFACE_TOKEN"),
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            use_cache=True
        )
        print("Creating pipeline...")
        
        # Create pipeline with optimizations
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
          #  do_sample=True,  # Enable sampling for better generation
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        
        # Initialize prompt template
        self.prompt_template = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a clinical expert. Use the following document excerpts to answer the question.
Answer in 1 or 2 concise sentences. If you don't know, simply say "I don't know." Do not repeat yourself.
<|eot_id|><|start_header_id|>user<|end_header_id|>

{context}

Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["context", "question"]
        )
        
        print("QA chain initialized successfully!")
        
    def search(self, query: str) -> list:
        """Search for relevant documents using the combined embeddings."""
        print(f"Searching for: {query}")
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            print("Query embedding generated")
            
            # Add zero padding for clinical part (K*128 dimensions)
            K = 5  # Same as in build_combined_store.py
            padded_query = np.concatenate([
                query_embedding,
                np.zeros(K * 128)
            ]).astype('float32')
            
            # Search in FAISS index with optimized parameters
            D, I = self.index.search(
                np.array([padded_query]),
                self.k  # Search for more documents than needed to filter
            )
            print(f"Found {len(I[0])} matches")
            
            # Get matching documents efficiently
            matches = []
            for i, (dist, idx) in enumerate(zip(D[0], I[0])):
                if idx < len(self.documents):  # Ensure index is valid
                    doc = self.documents[idx]
                    print(f"\nMatch {i+1}:")
                    print(f"Distance: {dist:.4f}")
                    print(f"Source: {doc.metadata.get('source', 'unknown')}")
                    print(f"Content preview: {doc.page_content[:100]}...")
                    
                    # Only include matches with reasonable similarity scores
                    if dist < 200.0:  # Increased threshold to allow more matches
                        matches.append({
                            'content': doc.page_content,
                            'metadata': {k:v for k,v in doc.metadata.items() if k != 'embedding'},
                            'score': float(dist),
                            'index': i
                        })
            
            # Sort matches by score and take the top k
            matches.sort(key=lambda x: x['score'])
            matches = matches[:self.k]
            
            return matches
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []
            
    def answer(
        self,
        question: str,
        chat_history: list = None
    ) -> dict:
        """Answer a question using the combined knowledge."""
        if chat_history is None:
            chat_history = []
        
        print("Searching for relevant documents...")
        # Search for relevant documents
        matches = self.search(question)
        
        if not matches:
            return {
                'answer': "I apologize, but I couldn't find any relevant information to answer your question. Please try rephrasing your question or ask about a different topic.",
                'sources': []
            }
        
        # Prepare context
        context = "\n\n".join([
            f"[{m['metadata']['source'].upper()}] {m['content']}"
            for m in matches
        ])
        print(f"Context length: {len(context)} characters")
        
        # Format prompt using template
        prompt = self.prompt_template.format(context=context, question=question)
        
        print("Generating answer...")
        print(f"Prompt length: {len(prompt)} characters")
        # Get answer from language model
        try:
            # Add more specific error handling and debugging
            print("Calling language model pipeline...")
            output = self.pipe(prompt)
            print(f"Pipeline output: {output}")
            
            if not output or len(output) == 0:
                raise ValueError("Empty response from language model")
                
            response = output[0]['generated_text']
            
            # Extract just the assistant's response
            try:
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            except:
                print("Could not extract assistant response, using full output")
            
            print("Answer generated successfully!")
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Detailed error in answer generation: {str(e)}")
            import traceback
            print(f"Stack trace: {traceback.format_exc()}")
            response = "I apologize, but I encountered an error while generating the answer. Please try again."
        
        return {
            'answer': response,
            'sources': matches
        } 