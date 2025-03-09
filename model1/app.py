from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch
from huggingface_hub import login
# import login_huggingface 
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define input schema using Pydantic
class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50  # Optional, default value
    num_return_sequences: int = 1  # Optional, default value

# # Hugging Face token (Set this as an environment variable in production)


# Retrieve the Hugging Face token from an environment variable
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")



# Load the model and pipeline during startup
@app.on_event("startup")

async def load_model():
    global pipe
    try:
        # Log in to Hugging Face
        if not HUGGINGFACE_TOKEN:
            raise ValueError("Hugging Face token is not set. Please set the HUGGINGFACE_TOKEN environment variable.")
        
        # print("Logging into Hugging Face...")
        # login(token=HUGGINGFACE_TOKEN, add_to_git_credential=False)  # Add explicit login
        # print("Logged in to Hugging Face successfully.")

        # print("Logging into Hugging Face...")
        # login(token=HUGGINGFACE_TOKEN, add_to_git_credential=True)
        
        print("Logging into Hugging Face...")
        login(token=HUGGINGFACE_TOKEN)

#  model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    #   model_id = "meta-llama/Llama-3.2-1B"
        # Load the Llama-3.2-1B model
        model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        pipe = pipeline(
            "text-generation", 
            model=model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"  # Automatically uses GPU if available
        )
        print("Model loaded successfully!")

        # Perform a local self-test
        print("Performing self-test...")
        result = pipe("what the capital of gharbia , egypt?!", max_length=30, num_return_sequences=1)
        print("Self-test successful! Generated text:", result[0]["generated_text"])
    except Exception as e:
        print(f"Error during model loading or self-test: {e}")
        raise RuntimeError("Could not load the model.")

# Define a POST endpoint for text generation
@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    try:
        # Run the pipeline with the provided prompt
        results = pipe(
            request.prompt, 
            max_length=request.max_length, 
            num_return_sequences=request.num_return_sequences
        )
        # Extract the generated text
        generated_texts = [result["generated_text"] for result in results]
        return {"generated_texts": generated_texts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Welcome to the Llama-3.2-1B text-generation API!"}