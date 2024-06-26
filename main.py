from fastapi import FastAPI, Request
from langchain_huggingface import HuggingFacePipeline


app = FastAPI()
summarizer = HuggingFacePipeline.from_model_id(
    model_id="facebook/bart-large-cnn",
    task="summarization",
)


@app.post("/summarize")
async def summarize(request: Request):
    data = await request.json()
    text = data.get("text", "")
    summary = await summarizer.ainvoke(text)
    return {"summary": summary}
