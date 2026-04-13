from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class SimpleRAGAgent:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        print("###Инициализация RAG агента###")
        if ollama_url is None:
            ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_url = ollama_url
        self.vectorstore = None
        self.llm = None
        self._load_knowledge_base()
        self._init_llm()
        print("Агент готов!\n")
    
    def _load_knowledge_base(self):
        """Загружает документы из папки data в векторную БД"""
        docs = []
        
        for f in os.listdir("data"):
            if f.endswith(".txt"):
                loader = TextLoader(f"data/{f}", encoding="utf-8")
                docs.extend(loader.load())
        

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma.from_documents(chunks, embeddings)
        print(f"###Загружено {len(chunks)} фрагментов###")
    
    def _init_llm(self):
        """Инициализация LLM"""
        
        self.llm = ChatOllama(
            base_url=self.ollama_url,
            model="llama3.2",
            temperature=0
        )
        print("LLM подключена (Ollama)")
     
    
    def _search_knowledge_base(self, question: str) -> str:
        """Поиск в базе знаний"""
        if self.vectorstore is None:
            return ""
        
        docs = self.vectorstore.similarity_search(question, k=5)
        
        return "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    def ask(self, question: str, history: list = None) -> str:
        """Основной метод - задать вопрос агенту"""
        
        if history is None:
            history = []
        
        # Проверяем, нужно ли уточнить страну
        q_lower = question.lower()
        has_country = any(c in q_lower for c in ["германия", "берлин", "germany", "франция", "париж", "france"])
        
        # Проверяем историю
        country_from_history = None
        for msg in history[-5:]:
            content = msg.get("content", "").lower()
            if "германия" in content or "берлин" in content:
                country_from_history = "Германия"
            elif "франция" in content or "париж" in content:
                country_from_history = "Франция"
        
        # Если нет страны и вопрос про чувствительные темы - уточняем
        needs_country = any(w in q_lower for w in ["стипендия", "виза", "налог", "рабочий день", "ставка"])
        if needs_country and not has_country and not country_from_history:
            return "Уточните, пожалуйста, страну: Германия или Франция?"
        
        # Определяем страну для поиска
        search_query = question
        if country_from_history and not has_country:
            search_query = f"{question} {country_from_history}"
        
        # Ищем в базе
        context = self._search_knowledge_base(search_query)
        
        # Формируем историю
        history_text = ""
        for msg in history[-6:]:
            history_text += f"{msg['role']}: {msg['content']}\n"
        
        # Формируем промпт
        prompt = f"""Ты консультант по международным стажировкам CdekStart.

ВОТ ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ (используй ТОЛЬКО её!):
{context if context else "ВНИМАНИЕ: Информация не найдена в базе!"}

История диалога:
{history_text}

Вопрос пользователя: {question}

ПРАВИЛА:
1. Используй ТОЛЬКО информацию из базы знаний выше
2. Если в базе есть конкретные цифры (евро, проценты, даты) - ОБЯЗАТЕЛЬНО их укажи
3. Если информации нет - скажи "Нет информации в базе знаний"
4. Не выдумывай ничего от себя

Ответ:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Ошибка: {e}"

_agent_instance = None

def get_agent(ollama_url: str = None):
    """Возвращает глобальный экземпляр агента"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SimpleRAGAgent(ollama_url=ollama_url)
    return _agent_instance