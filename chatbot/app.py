from fastapi import FastAPI
import uvicorn
import torch
from transformers import pipeline

app = FastAPI()

@app.get("/")
async def home():
  return {"message": "Chatbot for RDAI AI in Production"}

@app.post("/chatbot")
async def data(data: str):

    input_text = data

    pipe = pipeline("text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.bfloat16,
                device_map="auto")

    messages = [
        {
            "role": "system",
            "content": "You are a chatbot with positive energy",
        },
        {"role": "user", "content": input_text},
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True)

    outputs = pipe(prompt,
                  max_new_tokens=256,
                  do_sample=True,
                  temperature=0.7,
                  top_k=50,
                  top_p=0.95)

    return outputs[0]["generated_text"]

if __name__ == "__main__":
  uvicorn.run("app:app", reload=True, port=8000, host="0.0.0.0")