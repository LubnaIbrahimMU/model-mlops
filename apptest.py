from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
import torch
from huggingface_hub import login
from dotenv import load_dotenv
import os
from unsloth import FastLanguageModel
import subprocess

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define input schema using Pydantic
class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 500  # Default for reasoning tasks
    num_return_sequences: int = 1
    temperature: float = 0.6  # DeepSeek recommends 0.6

# Retrieve the Hugging Face token from an environment variable
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

@app.on_event("startup")
async def load_model():
    global pipe, tokenizer, model
    try:
        print("🔹 Checking GPU status before loading model:")
        subprocess.run(["nvidia-smi"])

        # Log in to Hugging Face
        if not HUGGINGFACE_TOKEN:
            raise ValueError("Hugging Face token is not set. Please set the HUGGINGFACE_TOKEN environment variable.")
        
        print("🔹 Logging into Hugging Face...")
        login(token=HUGGINGFACE_TOKEN)

        # Load model
        model_id = "unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit"

        print("🔹 Loading model... (this may take some time)")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_id,
            max_seq_length=4096,
            torch_dtype=torch.float16,  # Ensure FP16
            load_in_4bit=True,          # Enable 4-bit quantization
            device_map="cuda"           # Force full GPU usage
        )

        # Set up generation config
        model.generation_config.max_new_tokens = 4096
        model.generation_config.temperature = 0.6
        model.generation_config.top_p = 0.95

        # Create the pipeline
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            device=0  # Explicitly set device
        )
        
        print("✅ Model loaded successfully!")

        # Perform a self-test
        test_prompt = "Explain how to solve this math problem: What is the sum of the first 100 positive integers?"
        result = pipe(test_prompt, max_length=200, num_return_sequences=1, temperature=0.6)
        print("✅ Self-test successful! Sample:", result[0]["generated_text"][:100] + "...")

    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        raise RuntimeError(f"Could not load the model: {e}")

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    try:
        # Generate text
        results = pipe(
            request.prompt, 
            max_length=min(4096, request.max_length + len(tokenizer.encode(request.prompt))),
            num_return_sequences=request.num_return_sequences,
            temperature=request.temperature,
            top_p=0.95
        )

        # Extract the generated text
        generated_texts = [result["generated_text"] for result in results]
        return {"generated_texts": generated_texts}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the DeepSeek-R1-Distill-Qwen-32B text-generation API!",
        "model_info": "This API uses the 4-bit quantized version of DeepSeek-R1-Distill-Qwen-32B optimized by Unsloth.",
        "usage_tips": "For mathematical problems, include 'Please reason step by step, and put your final answer within \\boxed{}'."
    }
