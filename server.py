#source /home/khwillis@ad.wisc.edu/research-bot/.venv/bin/activate
#python -m pip install -r /home/khwillis@ad.wisc.edu/research-bot/requirements.txt
#python /home/khwillis@ad.wisc.edu/research-bot/server.py

#stop : fuser -k 51174/tcp

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
from pydantic import BaseModel
import os

from rag.chatbot import Chatbot
from rag.tools import tool_dict

# App setup
app = FastAPI(title="Genetic QTL Research Assistant")

# CORS (mostly unnecessary for same-origin, but safe defaults)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Static and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Provide a Flask-like url_for helper expected by the template
# The template uses: {{ url_for('static', filename='js/chat.js') }}
# Starlette expects 'path' instead of 'filename', so we shim it.
def _flask_style_url_for(endpoint: str, filename: str | None = None, **kwargs) -> str:
	if endpoint == "static" and filename:
		return f"/static/{filename}"
	return "/"


class ChatRequest(BaseModel):
	message: str


# Singleton chatbot to keep conversation context
MODEL_NAME = os.getenv("CHAT_MODEL", "qwen3:8b")
chatbot = Chatbot(MODEL_NAME, tool_dict)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
	return templates.TemplateResponse(
		"index.html",
		{"request": request, "url_for": _flask_style_url_for},
	)


@app.get("/healthz")
async def healthz():
	return {"status": "ok"}


@app.get("/stats")
async def stats():
	# Minimal demo stats; adjust to real data if desired
	return {
		"total_qtls": 500,
		"genes": ["GeneA"] * 303,
		"chromosomes": list(range(1, 20)),
	}


@app.post("/chat")
async def chat(req: ChatRequest):
	user_message = (req.message or "").strip()
	if not user_message:
		return JSONResponse({"response": "Please enter a message."}, status_code=400)
	try:
		answer = chatbot.chat(user_message)
		return {"response": answer}
	except Exception as exc:
		return JSONResponse({"response": f"Error: {exc}"}, status_code=500)


if __name__ == "__main__":
	import uvicorn
	port = int(os.getenv("PORT", "51174"))
	uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False) 