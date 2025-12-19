import os
import re
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from rag_config import VECTOR_DB_DIR

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================
# Модели данных
# =====================
class Question(BaseModel):
    question: str

class Source(BaseModel):
    source: Optional[str]
    snippet: str
    relevance: Optional[float] = None

class Answer(BaseModel):
    answer: str
    sources: List[Source]
    priority: str  # low, medium, high
    route_to: Optional[str] = None  # L1 / L2 / L3
    judge_reason: str
    confidence: Optional[float] = None

# =====================
# Утилиты
# =====================
STOP_WORDS = {
    "что", "как", "какие", "для", "чего", "это", "про", "о",
    "и", "или", "а", "в", "на", "по", "из", "ли", "же", "не",
    "но", "за", "у", "от", "до", "без", "под", "над", "при",
    "к", "с", "со", "во", "об", "то", "так", "вот", "тут"
}

HIGH_PRIORITY_TERMS = {
    "ошибка", "не работает", "критический", "сбой", "утечка", 
    "fraud", "компрометация", "банкротство", "мошенничество",
    "блокировка", "взлом", "атака", "уязвимость", "потеря данных",
    "несанкционированный", "кража", "взлом", "ddos"
}

MEDIUM_PRIORITY_TERMS = {
    "проблема", "недоступно", "ошибка системы", "баг", "глюк",
    "медленно", "тормозит", "не открывается", "не загружается",
    "сбой системы", "технические работы", "обслуживание"
}

def extract_core_terms(text: str) -> set:
    """Извлекает ключевые термины из текста"""
    words = re.findall(r"[a-zа-яё0-9]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in STOP_WORDS}

def is_linkable(source_path: str) -> bool:
    """Определяет, можно ли создать ссылку на источник"""
    if not source_path:
        return False
    ext = os.path.splitext(source_path)[1].lower()
    return ext in [".html", ".pdf", ".txt"]

def get_question_priority_keywords(question: str) -> str:
    """Определяет приоритет по ключевым словам"""
    q_lower = question.lower()
    
    if any(term in q_lower for term in HIGH_PRIORITY_TERMS):
        return "high"
    
    if any(term in q_lower for term in MEDIUM_PRIORITY_TERMS):
        return "medium"
    
    return "low"

def context_supports_question(context: str, question: str, min_matches: int = 2) -> bool:
    """Проверяет, содержит ли контекст ключевые термины вопроса"""
    context_lower = context.lower()
    terms = extract_core_terms(question)
    
    if not terms:
        return True
    
    # Считаем совпадения
    matches = sum(1 for term in terms if term in context_lower)
    return matches >= min_matches

# =====================
# Промпты
# =====================
PRIORITY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты — система классификации обращений в службу поддержки банка.
Определи критичность вопроса.

HIGH (высокий приоритет) — риск денег, безопасности, персональных данных, регуляторные риски, мошенничество, сбои в торговых системах, доступ к чужим данным.

MEDIUM (средний приоритет) — технические проблемы, ошибки в системах, вопросы по KYC, проблемы с доступом, блокировки аккаунтов.

LOW (низкий приоритет) — справка, инструкции, обучение, общие вопросы, документация.

Ответь ОДНИМ словом: LOW, MEDIUM или HIGH."""
    ),
    ("human", "ВОПРОС: {question}")
])

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты — AI-судья службы поддержки банка.

Твоя задача — определить:
1. Помог ли ответ пользователю
2. Насколько критичен вопрос
3. Нужно ли эскалировать обращение

ПРАВИЛА ОЦЕНКИ:

1. Если ответ содержит полезную информацию, шаги или объяснение,
   которые МОГУТ помочь пользователю — считай, что helped = true.
   Идеальный ответ НЕ требуется.

2. Если ответ отсутствует, содержит "информации нет" или не даёт пользы —
   helped = false.

3. Эскалация требуется ТОЛЬКО если helped = false.

ОЦЕНКА КРИТИЧНОСТИ:

HIGH / CRITICAL:
- деньги, финансы, переводы
- мошенничество, безопасность
- блокировки аккаунтов
- утечки данных
- регуляторные риски
- доступ к чужим данным
- сбои в торговых системах

MEDIUM:
- ошибки систем
- проблемы с доступом
- интеграции, API
- базы данных
- инфраструктура
- CRM, KYC
- технические проблемы

LOW:
- справка, информация
- инструкции, обучение
- общие вопросы
- документация

МАРШРУТИЗАЦИЯ:
- LOW  → L1
- MEDIUM → L2
- HIGH / CRITICAL → L3

Верни СТРОГО JSON:

{
  "helped": true | false,
  "priority": "low" | "medium" | "high",
  "route_to": "L1" | "L2" | "L3" | null,
  "reason": "краткое обоснование"
}

Если helped = true → route_to = null."""
    ),
    (
        "human",
        """ВОПРОС:
{question}

ОТВЕТ АГЕНТА:
{answer}

КОНТЕКСТ (если был):
{context}"""
    )
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты — AI-агент поддержки внутренних сервисов Сбербанка. 
Твоя задача — точно и полезно отвечать на вопросы сотрудников.

ИСПОЛЬЗУЙ ТОЛЬКО ИНФОРМАЦИЮ ИЗ КОНТЕКСТА!

ПРАВИЛА:
1. Отвечай на основе предоставленного контекста
2. Если информации недостаточно — так и скажи, но предложи возможные шаги
3. Будь конкретен: называй номера телефонов, ссылки, названия систем
4. Форматируй ответ: используй списки, выделяй важное
5. Если вопрос требует эскалации — так и скажи

СТИЛЬ:
- Профессионально, но дружелюбно
- Без излишней формальности
- С указанием конкретных действий"""
    ),
    (
        "human",
        """КОНТЕКСТ:
{context}

ВОПРОС:
{question}

ОТВЕТ (на основе контекста):"""
    )
])

EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Перефразируй вопрос 2-3 способами для улучшения поиска. 
Сохрани смысл, не добавляй новые факты.
Верни только варианты перефразировки, каждый с новой строки."""
    ),
    ("human", "ВОПРОС: {question}")
])

# =====================
# Основные функции
# =====================
def detect_priority(llm, question: str) -> str:
    """Определяет приоритет вопроса с помощью LLM"""
    try:
        result = (PRIORITY_PROMPT | llm | StrOutputParser()).invoke(
            {"question": question}
        )
        result = result.strip().upper()
        if result in {"LOW", "MEDIUM", "HIGH"}:
            return result
    except Exception as e:
        logger.warning(f"Ошибка при определении приоритета: {e}")
    
    # Fallback на keyword-based приоритет
    return get_question_priority_keywords(question).upper()

def judge_answer(llm, question: str, answer: str, context: str) -> dict:
    """Оценивает качество ответа"""
    parser = JsonOutputParser()
    chain = JUDGE_PROMPT | llm | parser
    
    try:
        result = chain.invoke({
            "question": question,
            "answer": answer,
            "context": context[:2000]  # Ограничиваем контекст
        })
        
        # Валидация результата
        if not isinstance(result, dict):
            raise ValueError("Некорректный формат ответа")
        
        if "helped" not in result:
            result["helped"] = True  # По умолчанию считаем, что помог
        
        if "priority" not in result:
            result["priority"] = "medium"
        
        if "route_to" not in result:
            result["route_to"] = None if result["helped"] else "L2"
        
        if "reason" not in result:
            result["reason"] = "Автоматическая оценка"
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при оценке ответа: {e}")
        # Fallback response
        return {
            "helped": True,
            "priority": "medium",
            "route_to": None,
            "reason": "Не удалось оценить ответ автоматически"
        }

def route_by_priority(priority: str) -> str:
    """Определяет маршрутизацию по приоритету"""
    priority_lower = priority.lower()
    if priority_lower == "high":
        return "L3"
    elif priority_lower == "medium":
        return "L2"
    else:
        return "L1"

def no_answer(priority: str) -> Answer:
    """Формирует ответ когда информации нет"""
    route = route_by_priority(priority)
    
    return Answer(
        answer=(
            "К сожалению, в моей базе знаний нет точной информации по этому вопросу. "
            "Рекомендую обратиться к специалисту поддержки для получения помощи."
        ),
        sources=[],
        priority=priority.lower(),
        route_to=None if priority == "LOW" else route,
        judge_reason="Ответ не найден в базе знаний",
        confidence=0.0
    )

def expand_query(llm, question: str) -> List[str]:
    """Расширяет запрос для улучшения поиска"""
    expansions = [question]  # Всегда включаем оригинальный вопрос
    
    try:
        expansions_raw = (EXPANSION_PROMPT | llm | StrOutputParser()).invoke(
            {"question": question}
        )
        
        # Разбираем результат на строки
        for line in expansions_raw.split('\n'):
            line = line.strip()
            if line and not line.startswith(("Вопрос:", "Перефразировка:", "Вариант:")):
                # Очищаем от маркеров списка
                line = re.sub(r'^[\d\-•\.\)\s]+', '', line)
                if line and len(line) > 10:
                    expansions.append(line)
        
        # Ограничиваем количество вариантов
        expansions = expansions[:4]
        
    except Exception as e:
        logger.warning(f"Ошибка при расширении запроса: {e}")
    
    logger.info(f"Расширенные запросы: {expansions}")
    return expansions

def format_sources(docs_with_scores, max_sources: int = 3) -> List[Source]:
    """Форматирует источники для ответа"""
    sources = []
    
    for doc, score in docs_with_scores[:max_sources]:
        snippet = doc.page_content[:400].strip()
        if len(doc.page_content) > 400:
            snippet += "..."
        
        source_path = doc.metadata.get("source", "")
        
        sources.append(Source(
            source=source_path if is_linkable(source_path) else None,
            snippet=snippet,
            relevance=1.0 - score  # Преобразуем расстояние в релевантность
        ))
    
    return sources

# =====================
# Приложение FastAPI
# =====================
def create_app() -> FastAPI:
    app = FastAPI(
        title="Sber Support RAG API",
        version="2.0.0",
        description="AI-агент поддержки внутренних сервисов Сбербанка",
    )
    
    # Инициализация компонентов
    logger.info("Инициализация эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    logger.info("Загрузка векторной базы...")
    try:
        vectordb = FAISS.load_local(
            str(VECTOR_DB_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Векторная база загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки векторной базы: {e}")
        raise RuntimeError(f"Не удалось загрузить векторную базу: {e}")
    
    logger.info("Инициализация LLM...")
    try:
        llm = ChatOllama(
            model="llama3.2",
            temperature=0.1,  # Низкая температура для более детерминированных ответов
            seed=42,
            timeout=30.0
        )
        logger.info("LLM инициализирована")
    except Exception as e:
        logger.error(f"Ошибка инициализации LLM: {e}")
        raise RuntimeError(f"Не удалось подключиться к Ollama: {e}. Убедитесь, что Ollama запущен и модель llama3.2 доступна.")
    
    @app.get("/")
    async def root():
        return {
            "service": "Sber Support RAG API",
            "version": "2.0.0",
            "status": "active",
            "endpoints": {
                "ask": "POST /ask - задать вопрос AI-агенту",
                "docs": "GET /docs - документация API"
            }
        }
    
    @app.post("/ask", response_model=Answer)
    async def ask(q: Question) -> Answer:
        """Основной endpoint для вопросов к AI-агенту"""
        question = q.question.strip()
        
        if not question or len(question) < 3:
            raise HTTPException(status_code=400, detail="Вопрос слишком короткий")
        
        logger.info(f"Вопрос: {question}")
        
        # 1. Определяем приоритет
        priority = detect_priority(llm, question)
        logger.info(f"Определен приоритет: {priority}")
        
        # 2. Поиск документов с расширением запроса
        all_docs_with_scores = []
        
        # Расширяем запрос
        expanded_queries = expand_query(llm, question)
        
        for query in expanded_queries:
            try:
                docs_scores = vectordb.similarity_search_with_score(query, k=8)
                all_docs_with_scores.extend(docs_scores)
                logger.debug(f"По запросу '{query}' найдено {len(docs_scores)} документов")
            except Exception as e:
                logger.warning(f"Ошибка поиска по запросу '{query}': {e}")
        
        # Удаляем дубликаты и сортируем по релевантности
        unique_docs = {}
        for doc, score in all_docs_with_scores:
            doc_hash = hash(doc.page_content[:200])  # Хэш начала для идентификации
            if doc_hash not in unique_docs or score < unique_docs[doc_hash][1]:
                unique_docs[doc_hash] = (doc, score)
        
        # Сортируем по возрастанию расстояния (меньше расстояние = больше релевантность)
        sorted_docs = sorted(unique_docs.values(), key=lambda x: x[1])
        
        if not sorted_docs:
            logger.warning("Документы не найдены")
            return no_answer(priority)
        
        logger.info(f"Найдено {len(sorted_docs)} уникальных документов")
        
        # 3. Формируем контекст из наиболее релевантных документов
        context_parts = []
        used_docs = []
        
        for doc, score in sorted_docs:
            # Проверяем релевантность (чем меньше score, тем лучше)
            if score > 0.9:  # Слишком низкая релевантность
                continue
            
            if not context_supports_question(doc.page_content, question):
                continue
            
            context_parts.append(doc.page_content)
            used_docs.append((doc, score))
            
            # Ограничиваем общий размер контекста
            if len('\n\n'.join(context_parts)) > 4000:
                break
            
            # Берем не более 5 документов
            if len(context_parts) >= 5:
                break
        
        if not context_parts:
            logger.warning("Не найдено релевантных документов")
            return no_answer(priority)
        
        context = "\n\n".join(context_parts)
        logger.info(f"Сформирован контекст из {len(context_parts)} документов, {len(context)} символов")
        
        # 4. Генерируем ответ на основе контекста
        try:
            answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
            answer_text = answer_chain.invoke({
                "context": context,
                "question": question
            }).strip()
            
            logger.info(f"Сгенерирован ответ: {answer_text[:100]}...")
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            answer_text = "Извините, произошла ошибка при обработке вашего вопроса."
        
        # 5. Оцениваем ответ
        judge_result = judge_answer(
            llm=llm,
            question=question,
            answer=answer_text,
            context=context[:1500]  # Ограничиваем для судьи
        )
        
        # 6. Формируем финальный ответ
        sources = format_sources(used_docs)
        
        # Рассчитываем уверенность на основе релевантности источников
        confidence = None
        if used_docs:
            # Средняя релевантность (1 - среднее расстояние)
            avg_score = sum(score for _, score in used_docs) / len(used_docs)
            confidence = max(0.0, min(1.0, 1.0 - avg_score))
        
        # Если ответ не помог, меняем текст ответа
        final_answer = answer_text
        if not judge_result.get("helped", True):
            final_answer = (
                "Я передал ваш вопрос специалисту поддержки. "
                "Ожидайте ответа в ближайшее время."
            )
            sources = []  # Не показываем источники если не помогли
        
        return Answer(
            answer=final_answer,
            sources=sources,
            priority=judge_result.get("priority", priority.lower()),
            route_to=judge_result.get("route_to"),
            judge_reason=judge_result.get("reason", "Автоматическая оценка"),
            confidence=confidence
        )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        try:
            # Проверяем доступность LLM
            test_response = llm.invoke("тест")
            return {
                "status": "healthy",
                "llm": "available",
                "vectordb": "loaded" if vectordb else "not loaded"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "llm": "unavailable"
            }
    
    return app

app = create_app()
