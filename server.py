"""EchoBuddy API. Run:  uvicorn server:app --reload   then open http://127.0.0.1:8000"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from emotion_map import CRISIS_RE, GROUPS

app = FastAPI(title="EchoBuddy API")
clf = None
if Path("model/config.json").exists():
    try:
        from transformers import pipeline
        clf = pipeline("text-classification", model="model", top_k=None,
                       function_to_apply="sigmoid")
    except Exception as e:  # falls back to keyword mode in the web page
        print("Model not loaded:", e)


class Msg(BaseModel):
    text: str


@app.post("/analyze")
def analyze(m: Msg):
    crisis = bool(CRISIS_RE.search(m.text))
    if clf is None:
        return {"engine": "keyword", "groups": {}, "crisis": crisis}
    out = clf(m.text)
    out = out[0] if isinstance(out[0], list) else out
    s = {d["label"]: d["score"] for d in out}
    groups = {g: round(max(s.get(l, 0.0) for l in ls), 3) for g, ls in GROUPS.items()}
    return {"engine": "distilbert", "groups": groups, "crisis": crisis}


@app.get("/")
def home():
    return FileResponse("index.html")
