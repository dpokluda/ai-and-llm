from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="LLM Model Server (Starter)")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 128

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenerateRequest):
    # TODO: wire to an actual LLM (OpenAI/Azure/HF) — placeholder echo for now.
    return {"output": f"ECHO: {req.prompt[:100]} ..."}
