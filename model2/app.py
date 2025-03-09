from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from dotenv import load_dotenv
import os
from unsloth import FastLanguageModel  # Add Unsloth import

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define input schema using Pydantic
class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 500  # Increased default value for reasoning tasks
    num_return_sequences: int = 1
    temperature: float = 0.6  # DeepSeek recommends 0.6 temperature

# Retrieve the Hugging Face token from an environment variable
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Load the model and pipeline during startup
@app.on_event("startup")
async def load_model():
    global pipe, tokenizer, model
    try:
        # Log in to Hugging Face
        if not HUGGINGFACE_TOKEN:
            raise ValueError("Hugging Face token is not set. Please set the HUGGINGFACE_TOKEN environment variable.")
        
        print("Logging into Hugging Face...")
        login(token=HUGGINGFACE_TOKEN)

        # Load the DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit model
        model_id = "unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit"
        

        # unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF



        
        # Using Unsloth's FastLanguageModel to load the quantized model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=4096,  # Set appropriate sequence length
            dtype=torch.bfloat16,
            load_in_4bit=True,    # Important to use 4-bit loading
            device_map="auto"     # Automatically use available GPUs
        )
        
        # Set up generation config according to the recommendations
        model.generation_config.max_new_tokens = 4096
        model.generation_config.temperature = 0.6  # Recommended temperature
        model.generation_config.top_p = 0.95      # Recommended top_p
        
        # Create the pipeline
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer
        )
        
        print("Model loaded successfully!")

        # Perform a local self-test
        print("Performing self-test...")
        test_prompt = "Explain how to solve this math problem: What is the sum of the first 100 positive integers?"
        result = pipe(test_prompt, max_length=200, num_return_sequences=1, temperature=0.6)
        print("Self-test successful! Generated text sample:", result[0]["generated_text"][:100] + "...")
    except Exception as e:
        print(f"Error during model loading or self-test: {e}")
        raise RuntimeError(f"Could not load the model: {e}")

# Define a POST endpoint for text generation
@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    try:
        # As recommended in the model card, avoid system prompts
        # and include any instructions directly in the user prompt
        results = pipe(
            request.prompt, 
            max_length=len(tokenizer.encode(request.prompt)) + request.max_length,
            num_return_sequences=request.num_return_sequences,
            temperature=request.temperature,
            top_p=0.95  # Recommended value
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
        "message": "Welcome to the DeepSeek-R1-Distill-Qwen-32B text-generation API!",
        "model_info": "This API uses the 4-bit quantized version of DeepSeek-R1-Distill-Qwen-32B optimized by Unsloth",
        "usage_tips": "For mathematical problems, include instructions like 'Please reason step by step, and put your final answer within \\boxed{}'."
    }