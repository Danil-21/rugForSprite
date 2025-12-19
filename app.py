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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================
# Константы
# =====================
CONFIDENCE_THRESHOLD = 0.4  # 70% порог уверенности

# =====================
# Модели данных
# =====================
class Question(BaseModel):
    question: str

class Source(BaseModel):
    source: Optional[str] = None
    snippet: str
    relevance: Optional[float] = None
    url: Optional[str] = None
    document_type: Optional[str] = None

class Answer(BaseModel):
    answer: str
    sources: List[Source]
    priority: str  # low, medium, high
    route_to: Optional[str] = None  # L1 / L2 / L3
    judge_reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_details: Optional[Dict] = None
    confidence_interpretation: Optional[str] = None
    confidence_below_threshold: bool = False  # Новое поле: уверенность ниже порога

# =====================
# Утилиты и конфигурация
# =====================
STOP_WORDS = {
    "что", "как", "какие", "для", "чего", "это", "про", "о",
    "и", "или", "а", "в", "на", "по", "из", "ли", "же", "не",
    "но", "за", "у", "от", "до", "без", "под", "над", "при",
    "к", "с", "со", "во", "об", "то", "так", "вот", "тут"
}

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

HIGH_PRIORITY_TERMS = {
    "деньги", "счет", "карта", "перевод", "платеж", "списание", "мошенничество",
    "взлом", "кража", "блокировка", "заблокирован", "недоступен", "ошибка перевода",
    "потерял", "украли", "несанкционированный", "арест", "арестован", "конфискация",
    "арест счета", "заморожен", "срочно", "экстренно", "критично", "угроза",
    "безопасность", "пароль", "вход", "взломали", "фишинг", "обман", "сняли",
    "пропали", "исчезли", "украден", "украдена", "заблокирована"
}

MEDIUM_PRIORITY_TERMS = {
    "не работает", "ошибка", "сбой", "технические проблемы", "не заходит",
    "приложение", "онлайн", "интернет-банк", "мобильное приложение", "сайт",
    "доступ", "авторизация", "вход", "логин", "пароль", "восстановление",
    "забыл", "утеря", "смена", "номер телефона", "email", "контакты",
    "настройки", "операция", "транзакция", "отказ", "отклонено", "не проходит",
    "не открывается", "зависает", "тормозит", "глючит", "баг"
}

def extract_core_terms(text: str) -> set:
    """Извлекает ключевые термины из текста"""
    words = re.findall(r"[a-zа-яё0-9]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in STOP_WORDS}

def get_sber_site_url(question: str) -> Optional[str]:
    """Определяет наиболее релевантный раздел сайта Сбера для вопроса"""
    question_lower = question.lower()
    
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
    
    urls_in_content = extract_urls_from_text(content[:1000])
    if urls_in_content:
        sber_urls = [url for url in urls_in_content if 'sberbank' in url]
        if sber_urls:
            return sber_urls[0]
    
    if source_path.endswith('.html'):
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

def get_relevancy_interpretation(score: float) -> str:
    """Интерпретация релевантности документа"""
    if score >= 0.9:
        return "Очень высокая релевантность"
    elif score >= 0.8:
        return "Высокая релевантность"
    elif score >= 0.7:
        return "Хорошая релевантность"
    elif score >= 0.6:
        return "Средняя релевантность"
    elif score >= 0.5:
        return "Умеренная релевантность"
    else:
        return "Низкая релевантность"

def analyze_answer_quality(answer: str, question: str) -> Dict:
    """Улучшенный анализ качества ответа - менее строгий к терминологии"""
    answer_lower = answer.lower()
    question_lower = question.lower()
    
    scores = {}
    
    # 1. Длина ответа (более гибкая)
    ideal_min_length = 30
    ideal_max_length = 500
    
    if len(answer) < ideal_min_length:
        length_score = len(answer) / ideal_min_length
    elif len(answer) > ideal_max_length:
        length_score = 1.0  # Длинные ответы - это хорошо
    else:
        length_score = 0.7 + (len(answer) - ideal_min_length) / (ideal_max_length - ideal_min_length) * 0.3
    
    length_score = max(0.3, min(1.0, length_score))
    scores["length"] = {
        "score": length_score,
        "length": len(answer),
        "ideal_range": f"{ideal_min_length}-{ideal_max_length}"
    }
    
    # 2. Отсутствие негативных фраз (самый важный фактор)
    negative_phrases = [
        "информации нет", "не знаю", "не могу ответить", 
        "нет данных", "не найдено", "неизвестно",
        "к сожалению, я не могу", "не удалось найти",
        "извините, но", "я не нашел"
    ]
    
    has_negative = any(phrase in answer_lower for phrase in negative_phrases)
    negative_score = 0.0 if has_negative else 1.0
    scores["negatives"] = {
        "score": negative_score,
        "has_negative": has_negative
    }
    
    # 3. Структура ответа
    structure_patterns = [
        r'\d+\.\s',  # Нумерованные списки
        r'[-•*]\s',  # Маркированные списки
        r'[Пп]ервый|[Вв]торой|[Тт]ретий',  # Порядковые номера
    ]
    
    structure_matches = sum(1 for pattern in structure_patterns 
                           if re.search(pattern, answer))
    structure_score = min(structure_matches / 2, 1.0)  # Нормализуем к 2 паттернам
    scores["structure"] = {
        "score": structure_score,
        "matches": structure_matches
    }
    
    # 4. Конкретность (менее строгая)
    if re.search(r'\d+', answer):  # Есть хоть какие-то цифры
        specificity_score = 0.8
    elif any(word in answer_lower for word in ['сбербанк', 'карта', 'счет', 'пароль']):
        specificity_score = 0.7
    else:
        specificity_score = 0.5
    
    scores["specificity"] = {
        "score": specificity_score
    }
    
    # Итоговый счет качества (веса)
    weights = {
        "length": 0.15,
        "negatives": 0.50,  # Самый важный - отсутствие "не знаю"
        "structure": 0.20,
        "specificity": 0.15
    }
    
    total_score = sum(scores[key]["score"] * weights[key] for key in weights)
    scores["total_score"] = total_score
    scores["weights"] = weights
    
    return scores

def calculate_qa_similarity(question: str, answer: str) -> float:
    """Рассчитывает семантическую схожесть вопроса и ответа"""
    stop_words = STOP_WORDS.union({
        'вас', 'вам', 'ваш', 'наш', 'свой', 'свои', 'своей',
        'этот', 'эта', 'это', 'эти', 'такой', 'такая', 'такое'
    })
    
    question_terms = set(
        word.lower() for word in re.findall(r'\b\w{3,}\b', question)
        if word.lower() not in stop_words
    )
    
    answer_terms = set(
        word.lower() for word in re.findall(r'\b\w{3,}\b', answer)
        if word.lower() not in stop_words
    )
    
    if not question_terms or not answer_terms:
        return 0.3
    
    intersection = len(question_terms.intersection(answer_terms))
    union = len(question_terms.union(answer_terms))
    
    if union == 0:
        return 0.0
    
    similarity = intersection / union
    
    important_terms = {'сбербанк', 'карта', 'счет', 'деньги', 'перевод', 'безопасность'}
    important_matches = len(important_terms.intersection(question_terms.intersection(answer_terms)))
    
    similarity += important_matches * 0.1
    return min(similarity, 1.0)

def calculate_confidence(
    docs_with_scores: List[Tuple],
    question: str,
    answer: str,
    context: str,
    priority: str
) -> Tuple[float, Dict, str]:
    """
    Улучшенный расчет уверенности - без строгого Q/A similarity
    
    Новые факторы:
    1. Релевантность лучшего документа (30%)
    2. Средняя релевантность топ-3 (20%)
    3. Качество ответа (35%) ← УВЕЛИЧЕНО
    4. Соответствие ответа контексту (15%) ← ВМЕСТО Q/A similarity
    """
    details = {
        "calculation_time": datetime.now().isoformat(),
        "factors": {},
        "question_preview": question[:100],
        "answer_length": len(answer),
        "calculation_method": "v3_context_alignment"
    }
    
    if not docs_with_scores:
        details["factors"]["no_documents"] = "Документы не найдены"
        interpretation = "Очень низкая уверенность (нет документов)"
        details["interpretation"] = interpretation
        return 0.1, details, interpretation
    
    # Анализируем документы
    relevancy_scores = [1.0 - score for _, score in docs_with_scores]
    
    # Фактор 1: Релевантность ЛУЧШЕГО документа (вес 30%)
    best_relevancy = max(relevancy_scores) if relevancy_scores else 0
    details["factors"]["best_document_relevancy"] = {
        "score": best_relevancy,
        "interpretation": get_relevancy_interpretation(best_relevancy)
    }
    relevancy_factor = best_relevancy * 0.3
    
    # Фактор 2: Средняя релевантность ТОП-3 документов (вес 20%)
    top_n = min(3, len(relevancy_scores))
    top_relevancy_scores = sorted(relevancy_scores, reverse=True)[:top_n]
    avg_top_relevancy = sum(top_relevancy_scores) / len(top_relevancy_scores)
    details["factors"]["top_documents_relevancy"] = {
        "average": avg_top_relevancy,
        "scores": top_relevancy_scores,
        "count": top_n
    }
    top_relevancy_factor = avg_top_relevancy * 0.2
    
    # Фактор 3: Качество и полнота ответа (вес 35%) ← УВЕЛИЧЕНО
    answer_quality_score = analyze_answer_quality(answer, question)
    details["factors"]["answer_quality"] = answer_quality_score
    quality_factor = answer_quality_score["total_score"] * 0.35
    
    # Фактор 4: Соответствие ответа контексту (вес 15%) ← НОВЫЙ ВМЕСТО Q/A similarity
    context_alignment = calculate_context_alignment(answer, context)
    details["factors"]["context_alignment"] = {
        "score": context_alignment,
        "method": "answer_terms_in_context"
    }
    alignment_factor = context_alignment * 0.15
    
    # Итоговая уверенность
    confidence_score = (
        relevancy_factor + 
        top_relevancy_factor + 
        quality_factor + 
        alignment_factor
    )
    
    # БОНУС: Если есть конкретные инструкции в ответе
    if has_concrete_instructions(answer):
        confidence_score = min(confidence_score + 0.1, 1.0)
        details["factors"]["concrete_instructions_bonus"] = 0.1
    
    # Ограничиваем диапазон и округляем
    confidence_score = max(0.0, min(1.0, confidence_score))
    confidence_score = round(confidence_score, 3)
    
    # Интерпретация
    interpretation = get_confidence_interpretation(confidence_score, priority)
    details["interpretation"] = interpretation
    details["final_score"] = confidence_score
    
    # Логируем факторы для отладки
    logger.info(f"📊 УВЕРЕННОСТЬ v3:")
    logger.info(f"   Лучшая релевантность: {best_relevancy:.2%}")
    logger.info(f"   Средняя топ-3: {avg_top_relevancy:.2%}")
    logger.info(f"   Качество ответа: {answer_quality_score['total_score']:.2%}")
    logger.info(f"   Соответствие контексту: {context_alignment:.2%}")
    logger.info(f"   ИТОГО: {confidence_score:.2%}")
    
    return confidence_score, details, interpretation


def calculate_context_alignment(answer: str, context: str) -> float:
    """
    Насколько хорошо ответ соответствует предоставленному контексту
    """
    if not answer or not context:
        return 0.5
    
    answer_lower = answer.lower()
    context_lower = context.lower()
    
    # Извлекаем ключевые термины из ответа (игнорируем стоп-слова)
    answer_terms = set(
        word for word in re.findall(r'\b\w{3,}\b', answer_lower)
        if word not in STOP_WORDS
    )
    
    if not answer_terms:
        return 0.5
    
    # Сколько терминов из ответа есть в контексте
    context_terms = set(
        word for word in re.findall(r'\b\w{3,}\b', context_lower[:2000])
        if word not in STOP_WORDS
    )
    
    matches = len(answer_terms.intersection(context_terms))
    alignment = matches / len(answer_terms)
    
    # Бонусы за качественные ответы
    bonuses = 0.0
    
    # 1. Бонус за инструкционные слова
    instruction_words = {'шаг', 'действие', 'необходимо', 'нужно', 'требуется', 
                        'можно', 'следует', 'рекомендуется', 'советуем'}
    if any(word in answer_lower for word in instruction_words):
        bonuses += 0.15
    
    # 2. Бонус за конкретные данные (номера, телефоны, суммы)
    if re.search(r'\d+', answer):
        bonuses += 0.10
    
    # 3. Бонус за ссылки или упоминание сайта
    if 'sberbank.ru' in answer_lower or 'https://' in answer_lower:
        bonuses += 0.10
    
    alignment = min(alignment + bonuses, 1.0)
    
    # Гарантируем минимальный уровень
    return min(max(alignment, 0.4), 1.0)  # Минимум 40%


def has_concrete_instructions(answer: str) -> bool:
    """Проверяет, содержит ли ответ конкретные инструкции"""
    answer_lower = answer.lower()
    
    # Паттерны конкретных инструкций
    patterns = [
        r'\d+\.\s',  # Нумерованные списки
        r'[-•*]\s',  # Маркированные списки
        r'шаг\s+\d+',  # Шаг 1, Шаг 2
        r'сначала\s+', r'затем\s+', r'после\s+',  # Последовательность
        r'нажмите\s+', r'выберите\s+', r'введите\s+',  # Конкретные действия
    ]
    
    return any(re.search(pattern, answer_lower) for pattern in patterns)


def get_confidence_interpretation(score: float, priority: str) -> str:
    """Интерпретация итоговой уверенности"""
    if score >= 0.85:
        base = "Очень высокая уверенность"
    elif score >= 0.70:
        base = "Высокая уверенность"
    elif score >= 0.55:
        base = "Средняя уверенность"
    elif score >= 0.40:
        base = "Умеренная уверенность"
    elif score >= 0.25:
        base = "Низкая уверенность"
    else:
        base = "Очень низкая уверенность"
    
    if priority == "high" and score > 0.7:
        return f"{base} (с учетом важности вопроса)"
    
    return base

def get_question_priority_keywords(question: str) -> str:
    """Определяет приоритет по ключевым словам"""
    q_lower = question.lower()
    
    if any(term in q_lower for term in HIGH_PRIORITY_TERMS):
        return "high"
    
    if any(term in q_lower for term in MEDIUM_PRIORITY_TERMS):
        return "medium"
    
    return "low"

def needs_immediate_escalation(confidence: float, priority: str) -> Tuple[bool, str]:
    """
    Определяет, требуется ли немедленная эскалация
    
    Возвращает:
    - нужно_ли_эскалировать (bool)
    - причина (str)
    """
    # Если уверенность ниже порога
    if confidence < CONFIDENCE_THRESHOLD:
        if priority == "high":
            return True, f"Уверенность {confidence:.1%} ниже порога {CONFIDENCE_THRESHOLD:.0%} при высоком приоритете вопроса"
        elif priority == "medium":
            return True, f"Уверенность {confidence:.1%} ниже порога {CONFIDENCE_THRESHOLD:.0%}"
        else:
            # Для low приоритета все равно эскалируем, но на L1
            return True, f"Уверенность {confidence:.1%} ниже порога {CONFIDENCE_THRESHOLD:.0%}"
    
    return False, ""

def get_escalation_level(priority: str, confidence: float) -> Tuple[str, str]:
    """
    Определяет уровень эскалации
    
    Возвращает:
    - уровень (L1/L2/L3)
    - причина
    """
    # Для HIGH приоритета - всегда L3, независимо от уверенности
    if priority == "high":
        if confidence < CONFIDENCE_THRESHOLD:
            return "L3", f"Критическая проблема с финансами/безопасностью. Уверенность {confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%}"
        else:
            return "L3", "Высокий приоритет вопроса (финансы/безопасность) требует экспертной проверки на L3"
    
    # Для остальных приоритетов - по уверенности
    if confidence < CONFIDENCE_THRESHOLD:
        if priority == "medium":
            return "L2", f"Техническая проблема требует специалиста. Уверенность {confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%}"
        else:  # low
            return "L1", f"Информационный вопрос требует уточнения. Уверенность {confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%}"
    
    # Если уверенность выше порога и не HIGH - маршрутизация не нужна
    return None, ""


def generate_low_confidence_response(priority: str, confidence: float, reason: str) -> Tuple[str, str]:
    """
    Генерирует ответ при низкой уверенности
    
    Возвращает:
    - ответ для клиента
    - уровень маршрутизации
    """
    if priority == "high":
        answer = (
            f"🔴 **СРОЧНО! ВАШ ВОПРОС ПЕРЕДАН СПЕЦИАЛИСТАМ БЕЗОПАСНОСТИ (L3)**\n\n"
            f"Ваш вопрос требует немедленного вмешательства специалистов по безопасности.\n\n"
            f"**НЕМЕДЛЕННЫЕ ДЕЙСТВИЯ:**\n"
            f"1. 📞 **Позвоните в службу безопасности Сбербанка: 900** (с мобильного) или **+7 (495) 500-55-50**\n"
            f"2. 🚫 **Немедленно заблокируйте карту** через мобильное приложение СберБанк Онлайн\n"
            f"3. 🏦 **Обратитесь в ближайшее отделение** с паспортом\n\n"
            f"**Информация об обращении:**\n"
            f"• Уровень обработки: **L3 (специалисты безопасности)**\n"
            f"• Время ответа: **в течение 15 минут**\n"
            f"• Телефон для срочных вопросов: **900**\n\n"
            f"*Причина эскалации: {reason}*"
        )
        return answer, "L3"
    
    elif priority == "medium":
        answer = (
            f"🔄 **ВАШ ВОПРОС ПЕРЕДАН ТЕХНИЧЕСКОМУ СПЕЦИАЛИСТУ (L2)**\n\n"
            f"Для решения вашего вопроса требуется помощь технического специалиста.\n\n"
            f"**Рекомендуемые действия:**\n"
            f"1. 📞 **Позвоните в техническую поддержку: 900**\n"
            f"2. 🌐 Посетите сайт: www.sberbank.ru\n"
            f"3. 📱 Используйте мобильное приложение\n\n"
            f"**Информация об обращении:**\n"
            f"• Уровень обработки: **L2 (технические специалисты)**\n"
            f"• Время ответа: **в течение 2 часов**\n"
            f"• Телефон поддержки: **900**\n\n"
            f"*Причина эскалации: {reason}*"
        )
        return answer, "L2"
    
    else:
        answer = (
            f"ℹ️ **ВАШ ВОПРОС ПЕРЕДАН КОНСУЛЬТАНТУ (L1)**\n\n"
            f"Для получения точной информации ваш вопрос передан консультанту.\n\n"
            f"**Вы можете:**\n"
            f"1. 📞 **Позвонить в справочную службу: 900**\n"
            f"2. 🌐 Найти информацию на сайте: www.sberbank.ru\n"
            f"3. 🏦 Обратиться в отделение банка\n\n"
            f"**Информация об обращении:**\n"
            f"• Уровень обработки: **L1 (консультанты)**\n"
            f"• Время ответа: **в течение 4 часов**\n"
            f"• Телефон справочной: **900**\n\n"
            f"*Причина эскалации: {reason}*"
        )
        return answer, "L1"

# =====================
# Промпты
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
        """Ты — AI-агент первой линии поддержки внутренних сервисов Сбербанка.
Твоя задача — помогать клиентам решать их вопросы, используя информацию из базы знаний.

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. Дай прямой ответ на вопрос без лишних приветствий
3. Если информации достаточно для решения проблемы — предоставь пошаговую инструкцию
4. Если в контексте есть URL на сайт Сбера - включи их в ответ
5. Если информации недостаточно — честно скажи что нужно передать вопрос специалисту
6. НИКОГДА не спрашивай "помог ли ответ" или "понятно ли объяснил"

СТИЛЬ ОБЩЕНИЯ:
- Профессионально, но без формальностей
- Только по делу
- Четко, конкретно, с нумерованными шагами при необходимости
- Без эмпатии и лишних слов (время ответа критично)

ФОРМАТ ОТВЕТА:
- Прямой ответ на вопрос
- Если нужны действия: 1. Сделай это. 2. Затем это. 3. Проверь то.
- Если нужно передать специалисту: "Для решения вопроса требуется подключение специалиста поддержки. Обращение создано."
"""
    ),
    (
        "human",
        """БАЗА ЗНАНИЙ ВНУТРЕННИХ СЕРВИСОВ:
{context}

ВОПРОС СОТРУДНИКА:
{question}

ТВОЙ ОТВЕТ:"""
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
            
            # Дополнительная проверка для критических фраз
            question_lower = question.lower()
            critical_phrases = [
                "украли", "потерял", "мошенничество", "взлом", "кража", 
                "списали", "несанкционирован", "пропали деньги", "украден",
                "заблокирова", "арест", "конфискация"
            ]
            
            if any(phrase in question_lower for phrase in critical_phrases):
                logger.info(f"⚠️ Обнаружена критическая фраза, повышаем приоритет до HIGH")
                return "HIGH"
            
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

def format_sources(docs_with_scores, max_sources: int = 3) -> List[Source]:
    """Форматирует источники для ответа клиенту с URL"""
    sources = []
    
    for doc, score in docs_with_scores[:max_sources]:
        snippet = doc.page_content[:300].strip()
        if len(doc.page_content) > 300:
            snippet += "..."
        
        source_path = doc.metadata.get("source", "")
        doc_type = doc.metadata.get("type", "document")
        
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
        "question": question[:200],
        "confidence": confidence,
        "interpretation": details.get("interpretation", "unknown"),
        "priority": details.get("priority", "unknown"),
        "factors": details.get("factors", {})
    }
    
    logger.info(f"📊 УВЕРЕННОСТЬ: {confidence:.2%} - {details.get('interpretation', 'unknown')}")
    
    try:
        log_file = "confidence_metrics.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Не удалось записать метрики в файл: {e}")

# =====================
# Приложение FastAPI
# =====================
def create_app() -> FastAPI:
    app = FastAPI(
        title="SberBank Client Support AI Agent",
        version="3.0.0",
        description="AI-агент первой линии поддержки для клиентов Сбербанка с оценкой уверенности",
    )
    
    # Инициализация компонентов
    logger.info("Инициализация AI-агента поддержки клиентов...")
    logger.info(f"Порог уверенности: {CONFIDENCE_THRESHOLD:.0%}")
    
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
            "version": "3.0.0",
            "status": "active",
            "confidence_threshold": f"{CONFIDENCE_THRESHOLD:.0%}",
            "logic": "Если уверенность < 70% → эскалация без ответа",
            "features": [
                "Оценка уверенности ответа",
                "Автоматическая эскалация при низкой уверенности",
                "Ссылки на сайт Сбербанка",
                "Умная маршрутизация по приоритету"
            ],
            "endpoints": {
                "ask": "POST /ask - задать вопрос от лица клиента",
                "health": "GET /health - проверка работоспособности",
                "confidence_stats": "GET /confidence - получить статистику уверенности"
            }
        }
    
    @app.get("/confidence")
    async def get_confidence_stats():
        """Получить статистику по уверенности агента"""
        try:
            log_file = "confidence_metrics.log"
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-100:]  # Последние 100 записей
                
                if not lines:
                    return {"message": "Нет данных о метриках"}
                
                metrics = [json.loads(line) for line in lines if line.strip()]
                
                confidences = [m.get("confidence", 0) for m in metrics]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Подсчет по интерпретациям
                interpretations = {}
                for m in metrics:
                    interpretation = m.get("interpretation", "unknown")
                    interpretations[interpretation] = interpretations.get(interpretation, 0) + 1
                
                # Подсчет по порогу
                below_threshold = len([c for c in confidences if c < CONFIDENCE_THRESHOLD])
                above_threshold = len([c for c in confidences if c >= CONFIDENCE_THRESHOLD])
                
                return {
                    "total_entries": len(metrics),
                    "average_confidence": f"{avg_confidence:.2%}",
                    "threshold": f"{CONFIDENCE_THRESHOLD:.0%}",
                    "below_threshold": below_threshold,
                    "above_threshold": above_threshold,
                    "below_threshold_percentage": f"{(below_threshold/len(metrics))*100:.1f}%" if metrics else "0%",
                    "interpretation_distribution": interpretations,
                    "recent_confidence_scores": confidences[-10:],
                    "recent_questions": [
                        {"question": m.get("question_preview", m.get("question", "N/A")[:50]),
                         "confidence": m.get("confidence", 0)}
                        for m in metrics[-5:]
                    ]
                }
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
        
        logger.info(f"🔍 ВОПРОС: '{question}'")
        
        # 1. Определяем приоритет
        priority = detect_priority(llm, question)
        logger.info(f"📊 Приоритет: {priority}")
        
        # 2. Расширяем запрос для лучшего поиска
        expanded_queries = expand_query(llm, question)
        
        # 3. Ищем релевантные документы
        all_docs_with_scores = []
        
        for query in expanded_queries:
            try:
                docs_scores = vectordb.similarity_search_with_score(query, k=10)
                all_docs_with_scores.extend(docs_scores)
                logger.debug(f"По запросу '{query}' найдено {len(docs_scores)} документов")
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
            logger.warning("❌ Документы не найдены в базе знаний")
            # НЕ создаем пустой ответ - сразу эскалация
            confidence = 0.1  # Минимальная уверенность
            confidence_details = {
                "calculation_time": datetime.now().isoformat(),
                "factors": {"no_documents": "Документы не найдены"},
                "interpretation": "Очень низкая уверенность (нет документов)"
            }
            
            # Всегда эскалируем если документов нет
            needs_escalation = True
            escalation_reason = "Информация по вопросу не найдена в базе знаний"
            
            escalation_level, level_reason = get_escalation_level(priority, confidence)
            answer_text, final_route = generate_low_confidence_response(
                priority, confidence, f"{escalation_reason}. {level_reason}"
            )
            
            return Answer(
                answer=answer_text,
                sources=[],
                priority=priority.lower(),
                route_to=final_route,
                judge_reason=f"Документы не найдены. {escalation_reason}",
                confidence=confidence,
                confidence_details=confidence_details,
                confidence_interpretation="Очень низкая уверенность (нет документов)",
                confidence_below_threshold=True
            )
        
        logger.info(f"📚 Найдено {len(sorted_docs)} документов")
        
        # 4. Формируем контекст из наиболее релевантных документов
        context_parts = []
        used_docs = []
        
        for doc, score in sorted_docs:
            if score > 0.95:  # Слишком низкая релевантность
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
            logger.warning("⚠️ Не найдено достаточно релевантных документов")
            # Всегда эскалируем если нет релевантных документов
            confidence = 0.2  # Очень низкая уверенность
            confidence_details = {
                "calculation_time": datetime.now().isoformat(),
                "factors": {"no_relevant_documents": "Не найдено релевантных документов"},
                "interpretation": "Очень низкая уверенность (нет релевантных документов)"
            }
            
            needs_escalation = True
            escalation_reason = "Не найдено релевантной информации по вашему вопросу"
            
            escalation_level, level_reason = get_escalation_level(priority, confidence)
            answer_text, final_route = generate_low_confidence_response(
                priority, confidence, f"{escalation_reason}. {level_reason}"
            )
            
            return Answer(
                answer=answer_text,
                sources=[],
                priority=priority.lower(),
                route_to=final_route,
                judge_reason=f"Недостаточно релевантных документов. {escalation_reason}",
                confidence=confidence,
                confidence_details=confidence_details,
                confidence_interpretation="Очень низкая уверенность (нет релевантных документов)",
                confidence_below_threshold=True
            )
        
        context = "\n\n".join(context_parts)
        logger.info(f"📝 Использовано {len(context_parts)} документов для формирования ответа")
        
        # 5. Генерируем ответ НА ОСНОВАНИИ КОНТЕКСТА
        try:
            answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
            answer_text = answer_chain.invoke({
                "context": context,
                "question": question
            }).strip()
            
            logger.info(f"🤖 Сгенерирован ответ длиной {len(answer_text)} символов")
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}")
            answer_text = "Извините, произошла ошибка при формировании ответа. Пожалуйста, обратитесь в поддержку по телефону 900."
        
        # 6. Рассчитываем уверенность в ответе
        confidence, confidence_details, interpretation = calculate_confidence(
            used_docs, question, answer_text, context, priority
        )
        
        # Логируем метрики уверенности
        log_confidence_metrics(question, confidence, confidence_details)
        
        # 7. Проверяем порог уверенности
        needs_escalation, escalation_reason = needs_immediate_escalation(confidence, priority)
        
        if needs_escalation:
            # 🔴 НИЗКАЯ УВЕРЕННОСТЬ - ЭСКАЛАЦИЯ БЕЗ ОТВЕТА
            logger.warning(f"🚨 НИЗКАЯ УВЕРЕННОСТЬ ({confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%}) - ЭСКАЛАЦИЯ")
            
            escalation_level, level_reason = get_escalation_level(priority, confidence)
            answer_text, final_route = generate_low_confidence_response(
                priority, confidence, escalation_reason
            )
            
            sources = []  # Не показываем источники при низкой уверенности
            
            return Answer(
                answer=answer_text,
                sources=sources,
                priority=priority.lower(),
                route_to=final_route,
                judge_reason=f"Низкая уверенность в ответе. {escalation_reason}",
                confidence=confidence,
                confidence_details=confidence_details,
                confidence_interpretation=interpretation,
                confidence_below_threshold=True
            )
        
        # 8. Уверенность ВЫШЕ порога - продолжаем нормальную обработку
        logger.info(f"✅ УВЕРЕННОСТЬ ВЫШЕ ПОРОГА ({confidence:.1%} >= {CONFIDENCE_THRESHOLD:.0%})")
        
        # 9. Оцениваем, насколько хорошо мы ответили и нужен ли сотрудник ПОСЛЕ ответа
        judge_result = judge_answer(
            llm=llm,
            question=question,
            answer=answer_text,
            context=context[:1500],
            priority=priority
        )
        
        helped = judge_result.get("helped", True)
        final_priority = judge_result.get("priority", priority.lower())
        
        # 10. Форматируем источники с URL
        sources = []
        if helped and used_docs:
            sources = format_sources(used_docs)
        
        # 11. Добавляем полезные ссылки к ответу
        if sources and any(source.url for source in sources):
            urls = [source.url for source in sources if source.url]
            if urls:
                links_text = "\n\n🔗 **Полезные ссылки:**\n"
                for i, url in enumerate(urls[:3], 1):
                    links_text += f"{i}. {url}\n"
                answer_text += links_text
        
        # 12. Определяем финальную маршрутизацию (если уверенность выше порога)
        route_to = judge_result.get("route_to")
        
        # Для high приоритета все равно маршрутизируем на L3
        if final_priority == "high" and not route_to:
            route_to = "L3"
            judge_result["reason"] = "Высокий приоритет вопроса требует проверки специалистом"
        
        if route_to:
            logger.info(f"🔄 Маршрутизация на {route_to} после ответа")
        else:
            logger.info("🎯 Агент полностью справился с вопросом")
        
        return Answer(
            answer=answer_text,
            sources=sources,
            priority=final_priority,
            route_to=route_to,
            judge_reason=judge_result.get("reason", "Автоматическая оценка"),
            confidence=confidence,
            confidence_details=confidence_details,
            confidence_interpretation=interpretation,
            confidence_below_threshold=False
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
                "confidence_threshold": f"{CONFIDENCE_THRESHOLD:.0%}",
                "logic": f"Эскалация если уверенность < {CONFIDENCE_THRESHOLD:.0%}",
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
    
    @app.get("/debug/confidence/{question}")
    async def debug_confidence(question: str):
        """Endpoint для отладки расчета уверенности"""
        try:
            # Ищем документы
            docs_with_scores = vectordb.similarity_search_with_score(question, k=5)
            
            # Генерируем тестовый ответ
            if docs_with_scores:
                context = "\n".join([doc.page_content for doc, _ in docs_with_scores[:3]])
                answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
                answer = answer_chain.invoke({"context": context, "question": question})
            else:
                context = ""
                answer = "Информация не найдена"
            
            # Определяем приоритет
            priority = detect_priority(llm, question)
            
            # Рассчитываем уверенность
            confidence, details, interpretation = calculate_confidence(
                docs_with_scores, question, answer, context, priority
            )
            
            return {
                "question": question,
                "priority": priority,
                "confidence": confidence,
                "interpretation": interpretation,
                "above_threshold": confidence >= CONFIDENCE_THRESHOLD,
                "threshold": CONFIDENCE_THRESHOLD,
                "details": details,
                "documents_found": len(docs_with_scores),
                "sample_documents": [
                    {
                        "relevancy": 1.0 - score,
                        "score": score,
                        "preview": doc.page_content[:200] + "..."
                    }
                    for doc, score in docs_with_scores[:3]
                ] if docs_with_scores else [],
                "answer_preview": answer[:500]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка отладки: {str(e)}")
    
    return app

app = create_app()
