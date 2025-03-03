from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import torch
import os
import psutil
import time

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def test_llama2():
    print("Testing Llama 3.2 3B Instruct model...")
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
    
    # Load environment variables
    load_dotenv()
    
    # Using Llama 3.2 3B Instruct model
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    temperature = 0.1
    max_new_tokens = 32  # Reduced for faster testing
    
    try:
        start_time = time.time()
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        print(f"Memory after loading tokenizer: {get_memory_usage():.2f} MB")
        
        # Load model with optimizations
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.getenv("HUGGINGFACE_TOKEN"),
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            use_cache=True
        )
        print(f"Memory after loading model: {get_memory_usage():.2f} MB")
        
        # Create pipeline with optimizations
        print("Creating pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,  # Enable sampling for better generation
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        
        # Test generation
        print("\nTesting text generation...")
        test_prompt = "<s>[INST] What is the role of HER-2/neu in breast cancer? [/INST]"
        response = pipe(test_prompt)[0]['generated_text']
        print("\nTest prompt:", test_prompt)
        print("\nGenerated response:", response)
        
        end_time = time.time()
        print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
        print(f"Final memory usage: {get_memory_usage():.2f} MB")
        print("\nModel test completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    test_llama2()