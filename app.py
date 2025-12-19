import os
import re
import json
import logging
from typing import List, Optional, Dict, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
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
    source: Optional[str] = None
    snippet: str
    relevance: Optional[float] = None
    url: Optional[str] = None  # Добавляем URL для ссылок
    document_type: Optional[str] = None

class Answer(BaseModel):
    answer: str
    sources: List[Source]
    priority: str  # low, medium, high
    route_to: Optional[str] = None  # L1 / L2 / L3
    judge_reason: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_details: Optional[Dict] = None  # Детали расчета уверенности

# =====================
# Утилиты и конфигурация
# =====================
STOP_WORDS = {
    "что", "как", "какие", "для", "чего", "это", "про", "о",
    "и", "или", "а", "в", "на", "по", "из", "ли", "же", "не",
    "но", "за", "у", "от", "до", "без", "под", "над", "при",
    "к", "с", "со", "во", "об", "то", "так", "вот", "тут"
}

# Сопоставление ключевых слов с разделами сайта Сбера
SBER_SITE_SECTIONS = {
    "карта": "https://www.sberbank.ru/ru/person/bank_cards",
    "вклад": "https://www.sberbank.ru/ru/person/contributions",
    "кредит": "https://www.sberbank.ru/ru/person/credits",
    "иис": "https://www.sberbank.ru/ru/person/investments/iis",
    "инвестиции": "https://www.sberbank.ru/ru/person/investments",
    "ипотека": "https://www.sberbank.ru/ru/person/credits/mortgage",
    "перевод": "https://www.sberbank.ru/ru/person/transfer",
    "платеж": "https://www.sberbank.ru/ru/person/payments",
    "онлайн": "https://www.sberbank.ru/ru/person/sberbankonline",
    "приложение": "https://www.sberbank.ru/ru/person/sberbankonline/mobileapp",
    "восстановление": "https://www.sberbank.ru/ru/person/sberbankonline/restore",
    "безопасность": "https://www.sberbank.ru/ru/person/security",
    "мошенничество": "https://www.sberbank.ru/ru/person/security/fraud",
    "отделение": "https://www.sberbank.ru/ru/person/branch",
    "банкомат": "https://www.sberbank.ru/ru/person/atm",
    "тариф": "https://www.sberbank.ru/ru/person/tariffs",
    "комиссия": "https://www.sberbank.ru/ru/person/tariffs",
    "faq": "https://www.sberbank.ru/ru/person/faq",
    "поддержка": "https://www.sberbank.ru/ru/person/help",
    "контакты": "https://www.sberbank.ru/ru/person/contacts",
}

# Базовые URL для документов
DOCUMENT_URL_MAPPING = {
    # Пример сопоставления имен файлов с URL
    "акция_инвестируй со сбербанком": "https://www.sberbank.ru/ru/person/investments/promotions",
    "восстановление доступа": "https://www.sberbank.ru/ru/person/sberbankonline/restore",
    "обыкновенная акция": "https://www.sberbank.ru/ru/person/investments/securities",
}

HIGH_PRIORITY_TERMS = {
    "деньги", "счет", "карта", "перевод", "платеж", "списание", "мошенничество",
    "взлом", "кража", "блокировка", "заблокирован", "недоступен", "ошибка перевода",
    "потерял", "украли", "несанкционированный", "арест", "арестован", "конфискация",
    "арест счета", "заморожен", "срочно", "экстренно", "критично", "угроза",
    "безопасность", "пароль", "вход", "взломали", "фишинг", "обман"
}

MEDIUM_PRIORITY_TERMS = {
    "не работает", "ошибка", "сбой", "технические проблемы", "не заходит",
    "приложение", "онлайн", "интернет-банк", "мобильное приложение", "сайт",
    "доступ", "авторизация", "вход", "логин", "пароль", "восстановление",
    "забыл", "утеря", "смена", "номер телефона", "email", "контакты",
    "настройки", "операция", "транзакция", "отказ", "отклонено", "не проходит"
}

LOW_PRIORITY_TERMS = {
    "как узнать", "где найти", "информация", "справка", "обучение",
    "инструкция", "руководство", "часто задаваемые", "faq", "вопрос",
    "ответ", "документ", "реквизиты", "адрес", "телефон", "график",
    "режим работы", "отделение", "банкомат", "условия", "тариф",
    "комиссия", "процент", "ставка", "кредит", "вклад", "инвестиции"
}

def extract_core_terms(text: str) -> set:
    """Извлекает ключевые термины из текста"""
    words = re.findall(r"[a-zа-яё0-9]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in STOP_WORDS}

def get_sber_site_url(question: str) -> Optional[str]:
    """Определяет наиболее релевантный раздел сайта Сбера для вопроса"""
    question_lower = question.lower()
    
    # Ищем ключевые слова в вопросе
    for keyword, url in SBER_SITE_SECTIONS.items():
        if keyword in question_lower:
            return url
    
    return None

def extract_urls_from_text(text: str) -> List[str]:
    """Извлекает URL из текста"""
    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*'
    urls = re.findall(url_pattern, text)
    return urls

def generate_document_url(source_path: str, content: str) -> Optional[str]:
    """Генерирует или извлекает URL для документа"""
    if not source_path:
        return None
    
    filename = os.path.basename(source_path).lower()
    
    # Пытаемся найти URL в самом документе
    urls_in_content = extract_urls_from_text(content[:1000])  # Проверяем начало
    if urls_in_content:
        # Фильтруем только URL Сбера
        sber_urls = [url for url in urls_in_content if 'sberbank' in url]
        if sber_urls:
            return sber_urls[0]
    
    # Пытаемся сопоставить по имени файла
    for doc_key, url in DOCUMENT_URL_MAPPING.items():
        if doc_key.lower() in filename:
            return url
    
    # Если это HTML файл, предполагаем что это страница сайта
    if source_path.endswith('.html'):
        # Извлекаем возможный URL из метаданных или имени файла
        html_name = os.path.splitext(filename)[0]
        for keyword, url in SBER_SITE_SECTIONS.items():
            if keyword in html_name:
                return url
    
    return None

def is_valid_sber_url(url: str) -> bool:
    """Проверяет, является ли URL допустимым URL Сбера"""
    if not url:
        return False
    
    valid_domains = ['sberbank.ru', 'sberbank.com', 'sber.ru']
    return any(domain in url for domain in valid_domains)

def context_supports_question(context: str, question: str, min_matches: int = 2) -> bool:
    """Проверяет, содержит ли контекст ключевые термины вопроса"""
    context_lower = context.lower()
    terms = extract_core_terms(question)
    
    if not terms:
        return True
    
    matches = sum(1 for term in terms if term in context_lower)
    return matches >= min_matches

def calculate_confidence(
    docs_with_scores: List[Tuple],
    question: str,
    answer: str,
    context: str
) -> Tuple[float, Dict]:
    """
    Рассчитывает уверенность агента в ответе
    
    Возвращает:
    - confidence_score (0.0-1.0)
    - confidence_details (детали расчета)
    """
    details = {
        "calculation_time": datetime.now().isoformat(),
        "factors": {}
    }
    
    if not docs_with_scores:
        details["factors"]["no_documents"] = "Документы не найдены"
        return 0.0, details
    
    # Фактор 1: Релевантность документов (вес 40%)
    relevancy_scores = [1.0 - score for _, score in docs_with_scores]
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
    details["factors"]["document_relevancy"] = {
        "average": avg_relevancy,
        "scores": relevancy_scores,
        "count": len(docs_with_scores)
    }
    
    relevancy_factor = avg_relevancy * 0.4
    
    # Фактор 2: Количество релевантных документов (вес 20%)
    good_docs = sum(1 for score in relevancy_scores if score > 0.7)
    doc_count_factor = min(good_docs / 3, 1.0) * 0.2
    details["factors"]["document_count"] = {
        "good_docs": good_docs,
        "total_docs": len(docs_with_scores),
        "factor": doc_count_factor
    }
    
    # Фактор 3: Совпадение ключевых терминов (вес 20%)
    question_terms = extract_core_terms(question)
    answer_terms = extract_core_terms(answer)
    
    if question_terms:
        term_overlap = len(question_terms.intersection(answer_terms)) / len(question_terms)
    else:
        term_overlap = 0.5
    
    term_factor = term_overlap * 0.2
    details["factors"]["term_overlap"] = {
        "question_terms": list(question_terms),
        "answer_terms": list(answer_terms),
        "overlap_ratio": term_overlap,
        "factor": term_factor
    }
    
    # Фактор 4: Качество ответа (вес 20%)
    answer_quality = 0.0
    answer_lower = answer.lower()
    
    # Положительные признаки
    positive_indicators = [
        len(answer) > 50,  # Ответ не слишком короткий
        not any(phrase in answer_lower for phrase in [
            "не знаю", "информации нет", "не могу ответить"
        ]),
        any(word in answer_lower for word in [
            "шаг", "инструкция", "необходимо", "требуется", "можно"
        ])
    ]
    
    answer_quality = sum(positive_indicators) / len(positive_indicators)
    quality_factor = answer_quality * 0.2
    details["factors"]["answer_quality"] = {
        "indicators": positive_indicators,
        "quality_score": answer_quality,
        "factor": quality_factor
    }
    
    # Итоговая уверенность
    confidence_score = relevancy_factor + doc_count_factor + term_factor + quality_factor
    
    # Ограничиваем диапазон
    confidence_score = max(0.0, min(1.0, confidence_score))
    
    # Добавляем интерпретацию
    if confidence_score > 0.8:
        interpretation = "Высокая уверенность"
    elif confidence_score > 0.6:
        interpretation = "Средняя уверенность"
    elif confidence_score > 0.3:
        interpretation = "Низкая уверенность"
    else:
        interpretation = "Очень низкая уверенность"
    
    details["interpretation"] = interpretation
    details["final_score"] = confidence_score
    
    return confidence_score, details

def get_question_priority_keywords(question: str) -> str:
    """Определяет приоритет по ключевым словам (для клиентов)"""
    q_lower = question.lower()
    
    if any(term in q_lower for term in HIGH_PRIORITY_TERMS):
        return "high"
    
    if any(term in q_lower for term in MEDIUM_PRIORITY_TERMS):
        return "medium"
    
    if any(term in q_lower for term in LOW_PRIORITY_TERMS):
        return "low"
    
    return "low"

# =====================
# Промпты (ориентированные на клиентов)
# =====================
PRIORITY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты — система классификации обращений клиентов в службу поддержки Сбербанка.
Определи критичность вопроса КЛИЕНТА.

HIGH (высокий приоритет) — всё, что связано с деньгами, счетами, картами, переводами, мошенничеством, безопасностью, блокировками, срочными проблемами, потерянными/украденными картами, несанкционированными списаниями.

MEDIUM (средний приоритет) — технические проблемы: не работает приложение, сайт, ошибки входа, проблемы с доступом, сбои в работе сервисов, вопросы по настройкам, восстановление доступа.

LOW (низкий приоритет) — информационные вопросы: справка, инструкции, режим работы, адреса отделений, тарифы, условия услуг, общие вопросы.

Ответь ОДНИМ словом: LOW, MEDIUM или HIGH."""
    ),
    ("human", "ВОПРОС КЛИЕНТА: {question}")
])

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты — AI-судья службы поддержки Сбербанка для клиентов.

Твоя задача — определить:
1. Насколько полно и полезно ответил агент
2. Нужна ли дополнительная помощь сотрудника ПОСЛЕ ответа агента

НОВЫЕ ПРАВИЛА:
1. Агент ВСЕГДА пытается ответить, если есть информация в документах
2. После ответа оцениваем: достаточно ли этого ответа или нужен сотрудник
3. Сотрудник нужен если:
   - Ответ неполный или непонятный
   - Требуются действия сотрудника (разблокировка, проверка операций)
   - Клиенту нужно общение с живым человеком
   - Вопрос слишком сложный для текстового ответа

Верни СТРОГО JSON:

{
  "helped": true | false,
  "priority": "low" | "medium" | "high",
  "route_to": "L1" | "L2" | "L3" | null,
  "reason": "краткое обоснование на русском"
}

Важно: route_to = null если агент полностью решил вопрос.
route_to != null если после ответа агента нужны действия сотрудника."""
    ),
    (
        "human",
        """ВОПРОС КЛИЕНТА:
{question}

ОТВЕТ АГЕНТА:
{answer}

КОНТЕКСТ (на основе которого отвечал агент):
{context}"""
    )
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты — AI-агент первой линии поддержки Сбербанка для клиентов.
Твоя задача — помогать клиентам решать их вопросы, используя информацию из базы знаний.

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. Будь максимально полезным и конкретным
3. Если в контексте есть URL на сайт Сбера - включи их в ответ
4. Если информации недостаточно - честно скажи
5. Дай максимально полный ответ из имеющейся информации

СТИЛЬ ОБЩЕНИЯ:
- Вежливо и профессионально
- Простыми словами
- С эмпатией
- Конкретно и по делу
- Если есть ссылки на сайт - добавь их в конце"""
    ),
    (
        "human",
        """ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ СБЕРБАНКА:
{context}

ВОПРОС КЛИЕНТА:
{question}

ТВОЙ ОТВЕТ КЛИЕНТУ (включи ссылки если они есть в контексте):"""
    )
])

EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Перефразируй вопрос клиента 2-3 способами для лучшего поиска в базе знаний.
Сохрани основной смысл, не добавляй новые факты.
Используй простые формулировки."""
    ),
    ("human", "ВОПРОС КЛИЕНТА: {question}")
])

# =====================
# Основные функции
# =====================
def detect_priority(llm, question: str) -> str:
    """Определяет приоритет вопроса клиента"""
    try:
        result = (PRIORITY_PROMPT | llm | StrOutputParser()).invoke(
            {"question": question}
        )
        result = result.strip().upper()
        if result in {"LOW", "MEDIUM", "HIGH"}:
            logger.info(f"LLM определил приоритет: {result}")
            return result
    except Exception as e:
        logger.warning(f"Ошибка при определении приоритета LLM: {e}")
    
    priority = get_question_priority_keywords(question).upper()
    logger.info(f"Keyword приоритет: {priority}")
    return priority

def judge_answer(llm, question: str, answer: str, context: str, priority: str) -> dict:
    """Оценивает ответ: помог ли агент и нужен ли сотрудник ПОСЛЕ ответа"""
    parser = JsonOutputParser()
    chain = JUDGE_PROMPT | llm | parser
    
    try:
        result = chain.invoke({
            "question": question,
            "answer": answer,
            "context": context[:2000]
        })
        
        if not isinstance(result, dict):
            result = {}
        
        answer_lower = answer.lower()
        if "helped" not in result:
            not_helpful_phrases = [
                "информации нет", "не знаю", "не могу ответить", 
                "нет данных", "не найдено информации"
            ]
            result["helped"] = not any(phrase in answer_lower for phrase in not_helpful_phrases)
        
        if "priority" not in result:
            result["priority"] = priority.lower()
        
        if "route_to" not in result:
            if not result.get("helped", True):
                if result["priority"] == "high":
                    result["route_to"] = "L3"
                elif result["priority"] == "medium":
                    result["route_to"] = "L2"
                else:
                    result["route_to"] = "L1"
            else:
                needs_human = (
                    result["priority"] == "high" and 
                    any(word in answer_lower for word in ["позвоните", "обратитесь", "сотрудник"])
                ) or (
                    result["priority"] == "medium" and
                    any(word in answer_lower for word in ["позвоните в поддержку", "обратитесь к оператору"])
                )
                
                if needs_human:
                    result["route_to"] = "L3" if result["priority"] == "high" else "L2"
                else:
                    result["route_to"] = None
        
        if "reason" not in result:
            if result.get("helped", True):
                if result.get("route_to"):
                    result["reason"] = "Агент дал информацию, но для дальнейших действий нужен сотрудник"
                else:
                    result["reason"] = "Агент полностью ответил на вопрос"
            else:
                result["reason"] = "Агент не смог найти информацию для ответа"
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при оценке ответа: {e}")
        return {
            "helped": True,
            "priority": priority.lower(),
            "route_to": "L2" if priority == "HIGH" else None,
            "reason": "Ошибка оценки, считаем что агент ответил"
        }

def needs_human_after_answer(question: str, answer: str, priority: str) -> bool:
    """Определяет, нужен ли сотрудник ПОСЛЕ того как агент ответил"""
    answer_lower = answer.lower()
    
    if priority == "high":
        if "как узнать" in question.lower() or "где найти" in question.lower():
            return False
        return True
    
    if priority == "medium":
        if any(phrase in answer_lower for phrase in [
            "позвоните", "обратитесь к", "свяжитесь с", "позвонить в поддержку"
        ]):
            return True
    
    return False

def generate_enhanced_answer(original_answer: str, question: str, priority: str, 
                           helped: bool, sources: List[Source]) -> str:
    """Улучшает ответ для клиента, добавляя информацию о маршрутизации если нужно"""
    
    # Собираем все уникальные URL из источников
    source_urls = []
    for source in sources:
        if source.url and is_valid_sber_url(source.url):
            source_urls.append(source.url)
    
    # Также получаем общий URL для вопроса
    question_url = get_sber_site_url(question)
    if question_url and question_url not in source_urls:
        source_urls.append(question_url)
    
    # Удаляем дубликаты
    source_urls = list(set(source_urls))
    
    enhanced_answer = original_answer
    
    # Добавляем ссылки если они есть
    if source_urls and helped:
        links_text = "\n\n🔗 **Полезные ссылки:**\n"
        for i, url in enumerate(source_urls[:3], 1):  # Ограничиваем 3 ссылками
            links_text += f"{i}. {url}\n"
        enhanced_answer += links_text
    
    if not helped:
        if priority == "high":
            enhanced_answer += (
                "\n\n🔴 Поскольку это срочный вопрос, связанный с безопасностью или деньгами, "
                "рекомендую НЕМЕДЛЕННО позвонить на горячую линию Сбербанка: 900."
            )
        elif priority == "medium":
            enhanced_answer += (
                "\n\nДля решения этого вопроса потребуется помощь специалиста поддержки. "
                "Пожалуйста, обратитесь в службу поддержки Сбербанка по телефону 900."
            )
    
    elif priority == "high" and needs_human_after_answer(question, original_answer, priority):
        enhanced_answer += (
            "\n\n⚠️ **После выполнения этих действий обязательно позвоните на горячую линию 900 "
            "для подтверждения и завершения процедуры.**"
        )
    
    elif priority == "medium" and needs_human_after_answer(question, original_answer, priority):
        enhanced_answer += (
            "\n\n📞 Если у вас остались вопросы или нужна дополнительная помощь, "
            "обратитесь в поддержку по телефону 900."
        )
    
    # Добавляем вежливое завершение
    import random
    endings = [
        "\n\nНадеюсь, эта информация была полезной!",
        "\n\nЕсли нужна дополнительная помощь - обращайтесь!",
        "\n\nЖелаю удачного дня!",
        "\n\nВсего доброго!"
    ]
    
    ending = random.choice(endings)
    return enhanced_answer + ending

def format_sources(docs_with_scores, max_sources: int = 3) -> List[Source]:
    """Форматирует источники для ответа клиенту с URL"""
    sources = []
    
    for doc, score in docs_with_scores[:max_sources]:
        snippet = doc.page_content[:300].strip()
        if len(doc.page_content) > 300:
            snippet += "..."
        
        source_path = doc.metadata.get("source", "")
        doc_type = doc.metadata.get("type", "document")
        
        # Генерируем URL для документа
        doc_url = generate_document_url(source_path, doc.page_content)
        
        if doc_type == "pdf":
            source_display = "Официальный документ Сбербанка"
        elif doc_type == "html":
            source_display = "Информация с сайта Сбербанка"
        else:
            source_display = "База знаний Сбербанка"
        
        sources.append(Source(
            source=source_display,
            snippet=snippet,
            relevance=1.0 - score,
            url=doc_url if is_valid_sber_url(doc_url) else None,
            document_type=doc_type
        ))
    
    return sources

def log_confidence_metrics(question: str, confidence: float, details: Dict):
    """Логирует метрики уверенности для аналитики"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "confidence": confidence,
        "details": details,
        "interpretation": details.get("interpretation", "unknown")
    }
    
    # Логируем в консоль
    logger.info(f"📊 Уверенность агента: {confidence:.2%} - {details.get('interpretation', 'unknown')}")
    logger.debug(f"Детали уверенности: {json.dumps(details, ensure_ascii=False, indent=2)}")
    
    # Также можно сохранять в файл для аналитики
    try:
        log_file = "confidence_metrics.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Не удалось записать метрики в файл: {e}")

def no_answer(priority: str, found_docs: bool = False) -> Answer:
    """Формирует ответ когда информации нет"""
    
    if found_docs:
        if priority == "HIGH":
            answer_text = (
                "Я нашел информацию по вашей теме, но не смог сформулировать точный ответ. "
                "Поскольку вопрос срочный, пожалуйста, немедленно позвоните на горячую линию: 900."
            )
            route = "L3"
        elif priority == "MEDIUM":
            answer_text = (
                "По вашему вопросу есть информация, но для точного решения требуется помощь специалиста. "
                "Обратитесь в поддержку по телефону 900."
            )
            route = "L2"
        else:
            answer_text = (
                "По вашему вопросу есть некоторая информация, но она неполная. "
                "Вы можете уточнить на сайте Сбербанка или позвонить по телефону 900."
            )
            route = "L1"
    else:
        if priority == "HIGH":
            answer_text = (
                "🔴 Срочный вопрос! Информации по вашему запросу нет в моей базе. "
                "Пожалуйста, НЕМЕДЛЕННО позвоните на горячую линию Сбербанка: 900."
            )
            route = "L3"
        elif priority == "MEDIUM":
            answer_text = (
                "Информации по вашему вопросу нет в моей базе знаний. "
                "Для решения проблемы обратитесь в службу поддержки по телефону 900."
            )
            route = "L2"
        else:
            answer_text = (
                "К сожалению, у меня нет информации по вашему вопросу. "
                "Вы можете найти подробности на сайте Сбербанка www.sberbank.ru "
                "или позвонить в справочную службу по телефону 900."
            )
            route = "L1"
    
    # Рассчитываем низкую уверенность для no_answer
    confidence, details = calculate_confidence([], question="", answer=answer_text, context="")
    
    # Логируем метрики
    log_confidence_metrics("", confidence, details)
    
    return Answer(
        answer=answer_text,
        sources=[],
        priority=priority.lower(),
        route_to=route,
        judge_reason="Информация не найдена или неполная в базе знаний",
        confidence=confidence,
        confidence_details=details
    )

def expand_query(llm, question: str) -> List[str]:
    """Расширяет запрос клиента для улучшения поиска"""
    expansions = [question]
    
    try:
        expansions_raw = (EXPANSION_PROMPT | llm | StrOutputParser()).invoke(
            {"question": question}
        )
        
        for line in expansions_raw.split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                clean_line = re.sub(r'^[\d\-•\.\)\s]+', '', line)
                if clean_line and clean_line != question:
                    expansions.append(clean_line)
        
        expansions = list(dict.fromkeys(expansions))[:4]
        
    except Exception as e:
        logger.warning(f"Ошибка при расширении запроса: {e}")
    
    return expansions

# =====================
# Приложение FastAPI
# =====================
def create_app() -> FastAPI:
    app = FastAPI(
        title="SberBank Client Support AI Agent",
        version="2.3.0",
        description="AI-агент первой линии поддержки для клиентов Сбербанка",
    )
    
    # Инициализация компонентов
    logger.info("Инициализация AI-агента поддержки клиентов...")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectordb = FAISS.load_local(
            str(VECTOR_DB_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Векторная база знаний загружена")
        
    except Exception as e:
        logger.error(f"Ошибка загрузки базы знаний: {e}")
        raise RuntimeError(f"Не удалось загрузить базу знаний: {e}")
    
    try:
        llm = ChatOllama(
            model="llama3.2",
            temperature=0.1,
            seed=42,
            timeout=30.0
        )
        logger.info("AI модель готова к работе с клиентами")
        
    except Exception as e:
        logger.error(f"Ошибка инициализации AI: {e}")
        raise RuntimeError(f"Не удалось подключиться к AI модели: {e}")
    
    @app.get("/")
    async def root():
        return {
            "service": "AI Agent - First Line Support for SberBank Clients",
            "version": "2.3.0",
            "status": "active",
            "features": [
                "Ответы на вопросы клиентов",
                "Автоматическая оценка уверенности",
                "Ссылки на сайт Сбербанка",
                "Умная маршрутизация",
                "Подробная аналитика"
            ],
            "endpoints": {
                "ask": "POST /ask - задать вопрос от лица клиента",
                "health": "GET /health - проверка работоспособности",
                "confidence_metrics": "GET /confidence - получить метрики уверенности"
            }
        }
    
    @app.get("/confidence")
    async def get_confidence_metrics():
        """Получить последние метрики уверенности"""
        try:
            log_file = "confidence_metrics.log"
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-50:]  # Последние 50 записей
                metrics = [json.loads(line) for line in lines]
                
                # Статистика
                if metrics:
                    confidences = [m.get("confidence", 0) for m in metrics]
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    return {
                        "total_entries": len(metrics),
                        "average_confidence": avg_confidence,
                        "recent_entries": metrics[-10:],  # Последние 10 записей
                        "interpretation_distribution": {
                            "high": len([m for m in metrics if m.get("confidence", 0) > 0.8]),
                            "medium": len([m for m in metrics if 0.6 < m.get("confidence", 0) <= 0.8]),
                            "low": len([m for m in metrics if 0.3 < m.get("confidence", 0) <= 0.6]),
                            "very_low": len([m for m in metrics if m.get("confidence", 0) <= 0.3])
                        }
                    }
                return {"message": "Нет данных о метриках"}
            return {"message": "Файл метрик не найден"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка получения метрик: {str(e)}")
    
    @app.post("/ask", response_model=Answer)
    async def ask(q: Question) -> Answer:
        """Основной endpoint для вопросов клиентов"""
        question = q.question.strip()
        
        if not question or len(question) < 3:
            raise HTTPException(
                status_code=400, 
                detail="Вопрос слишком короткий. Пожалуйста, опишите вашу проблему подробнее."
            )
        
        logger.info(f"Вопрос от клиента: {question}")
        
        # 1. Определяем приоритет
        priority = detect_priority(llm, question)
        logger.info(f"Определен приоритет: {priority}")
        
        # 2. Расширяем запрос
        expanded_queries = expand_query(llm, question)
        
        # 3. Ищем релевантные документы
        all_docs_with_scores = []
        
        for query in expanded_queries:
            try:
                docs_scores = vectordb.similarity_search_with_score(query, k=10)
                all_docs_with_scores.extend(docs_scores)
            except Exception as e:
                logger.warning(f"Ошибка поиска по запросу '{query}': {e}")
        
        # Удаляем дубликаты
        unique_docs = {}
        for doc, score in all_docs_with_scores:
            content_start = doc.page_content[:200]
            doc_hash = hash(content_start)
            if doc_hash not in unique_docs or score < unique_docs[doc_hash][1]:
                unique_docs[doc_hash] = (doc, score)
        
        sorted_docs = sorted(unique_docs.values(), key=lambda x: x[1])
        
        if not sorted_docs:
            logger.warning("Документы не найдены в базе знаний")
            return no_answer(priority, found_docs=False)
        
        logger.info(f"Найдено {len(sorted_docs)} документов")
        
        # 4. Формируем контекст
        context_parts = []
        used_docs = []
        
        for doc, score in sorted_docs:
            if score > 0.95:
                continue
            
            if not context_supports_question(doc.page_content, question, min_matches=1):
                continue
            
            context_parts.append(doc.page_content)
            used_docs.append((doc, score))
            
            if len('\n\n'.join(context_parts)) > 3500:
                break
            
            if len(context_parts) >= 5:
                break
        
        if not context_parts:
            logger.warning("Не найдено достаточно релевантных документов")
            return no_answer(priority, found_docs=True)
        
        context = "\n\n".join(context_parts)
        logger.info(f"Использовано {len(context_parts)} документов для формирования ответа")
        
        # 5. Генерируем ответ
        try:
            answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
            answer_text = answer_chain.invoke({
                "context": context,
                "question": question
            }).strip()
            
            logger.info(f"Сгенерирован ответ на основе контекста")
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            answer_text = "Извините, произошла ошибка при формировании ответа. Пожалуйста, обратитесь в поддержку по телефону 900."
        
        # 6. Рассчитываем уверенность
        confidence, confidence_details = calculate_confidence(
            used_docs, question, answer_text, context
        )
        
        # Логируем метрики уверенности
        log_confidence_metrics(question, confidence, confidence_details)
        
        # 7. Оцениваем ответ
        judge_result = judge_answer(
            llm=llm,
            question=question,
            answer=answer_text,
            context=context[:1500],
            priority=priority
        )
        
        helped = judge_result.get("helped", True)
        final_priority = judge_result.get("priority", priority.lower())
        
        # 8. Форматируем источники с URL
        sources = format_sources(used_docs)
        
        # 9. Улучшаем ответ для клиента (добавляем ссылки и маршрутизацию)
        final_answer = generate_enhanced_answer(
            answer_text, question, final_priority.upper(), helped, sources
        )
        
        # 10. Маршрутизация
        route_to = judge_result.get("route_to")
        
        if route_to:
            logger.info(f"После ответа требуется маршрутизация на {route_to}")
        else:
            logger.info("Агент полностью справился с вопросом")
        
        # 11. Выводим детали уверенности в лог (для защиты проекта)
        logger.info(f"📊 ДЕТАЛИ УВЕРЕННОСТИ для вопроса: '{question}'")
        logger.info(f"   Оценка уверенности: {confidence:.2%}")
        logger.info(f"   Интерпретация: {confidence_details.get('interpretation', 'N/A')}")
        logger.info(f"   Использовано документов: {len(used_docs)}")
        logger.info(f"   Средняя релевантность: {confidence_details.get('factors', {}).get('document_relevancy', {}).get('average', 0):.2%}")
        
        return Answer(
            answer=final_answer,
            sources=sources,
            priority=final_priority,
            route_to=route_to,
            judge_reason=judge_result.get("reason", "Автоматическая оценка"),
            confidence=confidence,
            confidence_details=confidence_details
        )
    
    @app.get("/health")
    async def health_check():
        """Проверка работоспособности сервиса"""
        try:
            test_response = llm.invoke("Привет")
            return {
                "status": "healthy",
                "service": "SberBank Client Support AI",
                "llm": "available",
                "vectordb": "loaded",
                "features": [
                    "confidence_calculation",
                    "url_linking", 
                    "smart_routing",
                    "detailed_analytics"
                ],
                "message": "Сервис готов к работе с клиентами"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "SberBank Client Support AI",
                "error": str(e),
                "llm": "unavailable",
                "message": "Требуется проверка подключения к Ollama"
            }
    
    return app

app = create_app()
