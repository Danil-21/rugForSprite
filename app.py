from typing import List, Optional
import re
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_config import VECTOR_DB_DIR

# =====================
# Models
# =====================
class Question(BaseModel):
    question: str

class Source(BaseModel):
    source: Optional[str]
    snippet: str

class Answer(BaseModel):
    answer: str
    sources: List[Source]

# =====================
# Utils
# =====================
STOP_WORDS = {
    "что", "как", "какие", "для", "чего", "это", "про", "о",
    "и", "или", "а", "в", "на", "по", "из", "ли", "же"
}

def extract_core_terms(text: str):
    words = re.findall(r"[a-zа-яё0-9]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in STOP_WORDS}

def context_supports_question(context: str, question: str) -> bool:
    """Мягкая проверка наличия ключевых слов/смысловых токенов в контексте"""
    context_l = context.lower()
    terms = extract_core_terms(question)
    if not terms:
        return True  # если ключевых слов нет — пропускаем
    return any(term in context_l for term in terms)

def no_answer() -> Answer:
    return Answer(
        answer=(
            "В моей базе знаний нет информации по этому вопросу. "
            "Рекомендую обратиться к специалисту."
        ),
        sources=[]
    )

# =====================
# App
# =====================
def create_app() -> FastAPI:
    app = FastAPI(title="Sber Support RAG", version="1.1.0-perdoc")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    vectordb = FAISS.load_local(
        str(VECTOR_DB_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    llm = ChatOllama(model="llama3.2", temperature=0.0, seed=42)

    @app.post("/ask", response_model=Answer)
    async def ask(q: Question) -> Answer:
        question = q.question.strip()

        # === 1. Semantic search по исходному вопросу ===
        docs_with_scores = vectordb.similarity_search_with_score(question, k=8)
        if not docs_with_scores:
            return no_answer()

        # === 2. Safe query expansion при низкой релевантности ===
        NEED_EXPANSION_THRESHOLD = 0.75
        if docs_with_scores[0][1] > NEED_EXPANSION_THRESHOLD:
            expansion_prompt = ChatPromptTemplate.from_messages([
                ("system",
                "Перефразируй вопрос 2 способами. "
                "Сохрани смысл, не добавляй новые факты. "
                "Верни только список вариантов."),
                ("human", "ВОПРОС: {question}")
            ])
            try:
                expansions_raw = (expansion_prompt | llm | StrOutputParser()).invoke({"question": question})
                expansions = [line.strip("-• ") for line in expansions_raw.splitlines() if line.strip()][:2]
            except Exception:
                expansions = []

            expanded_results = []
            for eq in expansions:
                expanded_results.extend(vectordb.similarity_search_with_score(eq, k=5))
            docs_with_scores = sorted(docs_with_scores + expanded_results, key=lambda x: x[1])

        # === 3. Собираем все релевантные документы (которые проходят проверку ключевых терминов) ===
        relevant_docs = []
        for doc, score in docs_with_scores:
            context = doc.page_content.strip()
            if not context:
                continue
            if context_supports_question(context, question):
                relevant_docs.append(doc)

        if not relevant_docs:
            return no_answer()

        # Объединяем контексты всех релевантных чанков
        full_context = "\n\n".join(d.page_content for d in relevant_docs)

        # === 4. Формируем prompt и получаем ответ ===
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Ты — AI-агент поддержки Сбербанка.\n"
                "Отвечай ТОЛЬКО буквальными фрагментами из переданного контекста.\n"
                "Не перефразируй, не добавляй примеры, слова или списки.\n"
                "Если ответа нет — скажи: информации нет."
            ),
            (
                "human",
                "КОНТЕКСТ:\n{context}\n\nВОПРОС: {question}\n\nОТВЕТ:"
            )
        ])
        chain = prompt | llm | StrOutputParser()
        try:
            answer_text = chain.invoke({"context": full_context, "question": question}).strip()
        except Exception:
            return no_answer()

        if answer_text:
            sources = [
                Source(
                    source=d.metadata.get("source", "Неизвестный источник"),
                    snippet=d.page_content[:500]
                )
                for d in relevant_docs
            ]
            return Answer(answer=answer_text, sources=sources)

        return no_answer()

    return app

app = create_app()
