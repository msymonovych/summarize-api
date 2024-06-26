import uvicorn
from fastapi import FastAPI, Request
from langchain_huggingface import HuggingFacePipeline


app = FastAPI()
summarizer = HuggingFacePipeline.from_model_id(
    model_id="facebook/bart-large-cnn",
    task="summarization",
    pipeline_kwargs={
        "max_length": 50,
        "min_length": 20
    }
)


@app.post("/summarize")
async def summarize(request: Request):
    data = await request.json()
    text = data.get("text", "")
    summary = await summarizer.ainvoke(text)
    return {"summary": summary}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
