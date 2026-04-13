from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from agent import get_agent

app = FastAPI()

sys.path.append(str(Path(__file__).parent.parent))

# Получаем URL Ollama из переменных окружения
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Получаем экземпляр агента с нужным URL
agent = get_agent(ollama_url=OLLAMA_URL)

# Путь к файлу сессий
DATA_FILE = "sessions.json"

# Путь к папке с шаблонами
TEMPLATES_DIR = "app/templates"

def load_all_sessions():
    """Загружает все сессии из одного JSON файла"""
    if not os.path.exists(DATA_FILE):
        return {}
    
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"Предупреждение: Файл {DATA_FILE} поврежден.")
        return {}

def save_all_sessions(sessions):
    """Сохраняет все сессии в один JSON файл"""
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Ошибка при сохранении: {e}")

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

@app.get("/")
async def root():
    """Главная страница - читает HTML из файла"""
    index_path = TEMPLATES_DIR +"/index.html"
    with open(index_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"Получен запрос: message={request.message}, session_id={request.session_id}")
    
    all_sessions = load_all_sessions()
    
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        print(f"Создана новая сессия: {session_id}")
    
    if session_id not in all_sessions:
        all_sessions[session_id] = []
    
    # Сохраняем вопрос пользователя
    all_sessions[session_id].append({
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now().isoformat()
    })
    
    # Получаем историю
    history_for_context = all_sessions[session_id][:]
    
    # Вызов агента
    answer = agent.ask(request.message, history_for_context)
    
    # Сохраняем ответ бота
    all_sessions[session_id].append({
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.now().isoformat()
    })
    
    save_all_sessions(all_sessions)
    
    return {"answer": answer, "session_id": session_id}

@app.get("/history/{session_id}")
def get_history(session_id: str):
    all_sessions = load_all_sessions()
    history = all_sessions.get(session_id, [])
    return {"history": history, "session_id": session_id}

@app.get("/sessions")
def list_sessions():
    all_sessions = load_all_sessions()
    sessions_list = []
    for session_id, history in all_sessions.items():
        sessions_list.append({
            "id": session_id,
            "messages_count": len(history),
            "created": history[0]["timestamp"] if history else None,
            "last_message": history[-1]["content"][:] if history else None
        })
    return {"sessions": sessions_list}

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    all_sessions = load_all_sessions()
    if session_id in all_sessions:
        del all_sessions[session_id]
        save_all_sessions(all_sessions)
        return {"deleted": session_id}
    return {"error": "Session not found"}

