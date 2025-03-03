from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re
from langchain.document_loaders import PyPDFLoader

class PDFCleaner:
    def __init__(self):
        """Initialize the PDF cleaner for research papers."""
        pass

    def clean_text(self, text):
        """Clean extracted text from PDF."""
        # Remove URLs and references to them
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove special characters and formatting artifacts
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Fix hyphenated words
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix line-break hyphens
        text = re.sub(r'(\w+)-\s*(\w+)', r'\1\2', text)  # Fix inline hyphens
        
        # Fix common OCR errors
        text = re.sub(r'[lI]ll\b', 'et al', text)  # Fix "Ill" -> "et al"
        text = re.sub(r'\bihill\b', 'ibid', text)  # Fix "ihill" -> "ibid"
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between camelCase
        text = re.sub(r'(?<=[a-zA-Z])\.(?=[a-zA-Z])', '. ', text)  # Add space after period
        
        # Fix scientific notation and measurements
        text = re.sub(r'([0-9]+)\s*[xX×]\s*10\s*[-−]?\s*([0-9]+)', r'\1e\2', text)  # Fix scientific notation
        text = re.sub(r'([0-9]+)\s*[.,]\s*([0-9]+)', r'\1.\2', text)  # Fix decimal points
        text = re.sub(r'([0-9]+)\s*([%°℃])', r'\1\2', text)  # Fix units
        text = re.sub(r'±\s*([0-9]+)', r'±\1', text)  # Fix plus-minus
        text = re.sub(r'([<>])\s*([0-9]+)', r'\1\2', text)  # Fix comparisons
        text = re.sub(r'([0-9]+)\s*-\s*([0-9]+)', r'\1-\2', text)  # Fix ranges
        
        # Clean up scientific paper specific artifacts
        text = re.sub(r'\b[A-Z]{2,}\b', lambda m: m.group(0).title(), text)  # Convert ALL CAPS to Title Case
        text = re.sub(r'\s*\([^)]*\d+[^)]*\)\s*', ' ', text)  # Remove reference citations
        text = re.sub(r'Fig\.\s*\d+[A-Za-z]?', '', text)  # Remove figure references
        text = re.sub(r'Table\s*\d+', '', text)  # Remove table references
        text = re.sub(r'et al\.?,?', 'et al', text)  # Standardize et al
        text = re.sub(r'i\.e\.?,?', 'i.e.', text)  # Standardize i.e.
        text = re.sub(r'e\.g\.?,?', 'e.g.', text)  # Standardize e.g.
        
        # Remove journal formatting
        text = re.sub(r'SCIENCE,?\s+VOL\.?\s*\d+.*?\d{4}', '', text)
        text = re.sub(r'Downloaded from.*?org', '', text)
        text = re.sub(r'on\s+[A-Z][a-z]+\s+\d+,\s+\d{4}', '', text)
        text = re.sub(r'pp?\.\s*\d+[-–]\d+', '', text)  # Remove page numbers
        
        # Clean up statistical notation
        text = re.sub(r'[pP]\s*[<=>]\s*0?\.\d+', '', text)  # Remove p-values
        text = re.sub(r'[nN]\s*=\s*\d+', '', text)  # Remove sample sizes
        text = re.sub(r'95%\s*CI', '95% confidence interval', text)  # Expand CI
        text = re.sub(r'(?<=[0-9])\s*%', '%', text)  # Fix percentages
        
        # Clean up whitespace and formatting
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'[^\w\s.,;?!-/]', '', text)  # Remove special characters but preserve forward slash
        text = re.sub(r'\s+([.,;?!])', r'\1', text)  # Remove spaces before punctuation
        text = re.sub(r'\.{2,}', '.', text)  # Replace multiple periods
        text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces
        
        # Remove isolated numbers and short strings
        text = re.sub(r'\b\d+\b(?!\s*[a-zA-Z])', '', text)  # Remove isolated numbers
        text = re.sub(r'\b[a-zA-Z]\b(?!\s*[a-zA-Z])', '', text)  # Remove single letters
        
        # Preserve important scientific terms and identifiers (moved to the end)
        text = re.sub(r'proto\s*oncogene', 'proto-oncogene', text, flags=re.IGNORECASE)
        text = re.sub(r'hormonal\s*receptor', 'hormonal-receptor', text, flags=re.IGNORECASE)
        text = re.sub(r'erb\s*B', 'erbB', text, flags=re.IGNORECASE)
        
        # Preserve HER-2/neu and related terms (most specific patterns first)
        # First, standardize any variations of HER-2/neu
        text = re.sub(r'Her\s*2\s*[/\\]\s*neu', 'HER-2/neu', text, flags=re.IGNORECASE)
        text = re.sub(r'Her\s*2', 'HER-2', text, flags=re.IGNORECASE)
        text = re.sub(r'HER\s*2\s*[/\\]\s*neu', 'HER-2/neu', text, flags=re.IGNORECASE)
        text = re.sub(r'HER\s*2', 'HER-2', text, flags=re.IGNORECASE)
        
        # Then handle specific terms
        text = re.sub(r'HER-2/neu\s*positive', 'HER-2/neu positive', text, flags=re.IGNORECASE)
        text = re.sub(r'HER-2/neu\s*negative', 'HER-2/neu negative', text, flags=re.IGNORECASE)
        text = re.sub(r'HER-2/neu\s*amplification', 'HER-2/neu amplification', text, flags=re.IGNORECASE)
        text = re.sub(r'HER-2/neu\s*overexpression', 'HER-2/neu overexpression', text, flags=re.IGNORECASE)
        
        # Fix any remaining variations
        text = re.sub(r'Her\s*/\s*neu', 'HER-2/neu', text, flags=re.IGNORECASE)
        text = re.sub(r'Her\s*2\s*/\s*neu', 'HER-2/neu', text, flags=re.IGNORECASE)
        text = re.sub(r'Her\s*2\s*/\s*neu', 'HER-2/neu', text, flags=re.IGNORECASE)
        
        return text.strip()

def process_pdf(pdf_path: str, chunk_size: int = 1500, chunk_overlap: int = 100):
    """Process a PDF file and return cleaned chunks.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of each text chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of Document objects with cleaned text
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
       # separators=["\n\n", ".", " ", ""]
    )
    
    # Initialize PDF cleaner
    pdf_cleaner = PDFCleaner()
    
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print("\nOriginal text from first page:")
        print(documents[0].page_content[:500] + "...")
        
        # Clean the text
        for doc in documents:
            # Basic cleaning
            doc.page_content = doc.page_content.replace('\n', ' ')
            doc.page_content = doc.page_content.replace('\xad', '')
            doc.page_content = doc.page_content.replace('erl,B', 'erbB')
            
            # Additional cleaning using PDFCleaner
            doc.page_content = pdf_cleaner.clean_text(doc.page_content)
        
        print("\nCleaned text from first page:")
        print(documents[0].page_content[:500] + "...")
        
        # Split into chunks
        chunks = text_splitter.split_documents(documents)
        print(f"\nCreated {len(chunks)} chunks from {pdf_path}")
        
        # Print first few chunks
        print("\nFirst 3 chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print("-" * 50)
            print(chunk.page_content)
            print("-" * 50)
        
        return chunks
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        raise 