"""
Modal Script for Deploying Laya API Endpoint
======================================================
This runs an ultra-fast FastAPI server on Modal to serve the fine-tuned model.

Usage:
  modal deploy serve_laya.py
"""

import modal
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create the Modal app
app = modal.App("laya-slop-api")

# Define the image with all dependencies
laya_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", 
        "transformers>=4.48.0", 
        "fastapi",
        "pydantic",
        "tiktoken",
        "sentencepiece",
        "tokenizers>=0.21.0"
    )
)

# Reference the permanent volume where we saved the model
model_volume = modal.Volume.from_name("laya-models-vol")

# Create FastAPI app
web_app = FastAPI()

# Allow CORS for the Chrome Extension (so LinkedIn can call it)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schema for the request
class PredictRequest(BaseModel):
    text: str

# Use a Modal Class to keep the model in memory across requests (warm starts)
@app.cls(
    image=laya_image, 
    gpu="T4",
    volumes={"/model_cache": model_volume},
    container_idle_timeout=120  # Keep container alive for 2 mins between requests
)
class LayaSlopDetector:
    @modal.enter()
    def load_model(self):
        print("Loading fine-tuned model into VRAM...")
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Path where we saved it in finetune_laya.py
        model_path = "/model_cache/laya-slop-final"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except ValueError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    @modal.method()
    def predict(self, text: str) -> float:
        import torch
        
        # 1. Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512, 
            padding=True
        ).to(self.device)
        
        # 2. Forward pass (No generation, just logits!)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 3. Softmax to get probability
            probs = torch.softmax(logits, dim=-1)
            
            # Assuming label 1 is "slop" and label 0 is "human"
            slop_prob = probs[0][1].item()
            
        return slop_prob

# Expose the class via an HTTP endpoint
@app.function(image=laya_image)
@modal.asgi_app()
def fastapi_app():
    # Instantiate the Modal class
    detector = LayaSlopDetector()

    @web_app.post("/predict")
    def predict_endpoint(req: PredictRequest):
        # Call the remote GPU method
        prob = detector.predict.remote(req.text)
        return {"slop_probability": prob}

    return web_app
