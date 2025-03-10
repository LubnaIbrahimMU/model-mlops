from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch import bfloat16
import torch
import transformers
from huggingface_hub import login
from dotenv import load_dotenv
import os
import subprocess

# Import Unsloth's optimized loading
from unsloth import FastLanguageModel

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define input schema using Pydantic
class TextGenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256  # Changed from max_length to match your original code
    num_return_sequences: int = 1
    temperature: float = 0.7  # Set to match your original code
    do_sample: bool = True  # Added to match your original code

# Retrieve the Hugging Face token from an environment variable
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

@app.on_event("startup")
async def load_model():
    global pipe, tokenizer, model
    try:
        print("🔹 Checking GPU status before loading model:")
        subprocess.run(["nvidia-smi"])
        
        # Print versions to ensure compatibility
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Transformers version: {transformers.__version__}")

        # Log in to Hugging Face
        if not HUGGINGFACE_TOKEN:
            raise ValueError("Hugging Face token is not set. Please set the HUGGINGFACE_TOKEN environment variable.")
        
        print("🔹 Logging into Hugging Face...")
        login(token=HUGGINGFACE_TOKEN)

        # Use the model you specified
        model_id = "unsloth/Qwen2.5-14B-Instruct-1M-unsloth-bnb-4bit"

        print("🔹 Loading model... (this may take some time)")

        # Using Unsloth's FastLanguageModel to load the quantized model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map="auto",  # Let it decide optimal device mapping
        )

        # Create a text generation pipeline
        from transformers import pipeline
        pipe = pipeline(
            model=model,
            tokenizer=tokenizer,
            task='text-generation',
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )
        
        print("✅ Model loaded successfully!")

        # Perform a local self-test
        print("🔹 Performing self-test...")
        test_prompt = "Explain the benefits of model quantization in simple terms:"
        result = pipe(test_prompt)
        print("✅ Self-test successful! Generated text sample:", result[0]["generated_text"][:100] + "...")

    except Exception as e:
        print(f"❌ Error during model loading or self-test: {e}")
        raise RuntimeError(f"Could not load the model: {e}")
    
# Define a POST endpoint for text generation
@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    try:
        # Generate text
        results = pipe(
            request.prompt, 
            max_new_tokens=request.max_new_tokens,
            num_return_sequences=request.num_return_sequences,
            temperature=request.temperature,
            do_sample=request.do_sample
        )

        # Extract the generated text
        generated_texts = [result["generated_text"] for result in results]
        return {"generated_texts": generated_texts}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

# Root endpoint for testing
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Qwen2.5-14B-Instruct text-generation API!",
        "model_info": "This API uses the 4-bit quantized version of Qwen2.5-14B-Instruct optimized by Unsloth",
        "usage_example": "Send a POST request to /generate with a JSON body containing your prompt"
    }