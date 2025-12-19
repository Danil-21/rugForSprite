import re
import logging
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from rag_config import DATA_RAW_DIR, VECTOR_DB_DIR

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Очистка текста от лишних элементов"""
    # Восстанавливаем переносы слов
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    
    patterns = [
        r"Стр\.\s*\d+\s+из\s+\d+",
        r"\d+\s+страница\s+из\s+\d+",
        r"©.*?(Сбербанк|Sberbank|20\d{2}|\d{4})",
        r"Конфиденциально|Для внутреннего использования|Версия\s*\d+",
        r"ID\s*документа[:\s]*[A-Z0-9\-]+",
        r"Тел\.:?\s*900|Телефон:\s*900",
        r"\s{2,}",
    ]
    
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    
    # Убираем множественные переносы строк
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def load_html_documents(path: Path) -> List[Document]:
    """Загружает HTML-файлы из папки"""
    docs = []
    
    for file in path.glob("*.html"):
        try:
            with open(file, encoding="utf-8") as f:
                html = f.read()
            soup = BeautifulSoup(html, "html.parser")
            
            # Удаляем ненужные элементы
            for tag in soup.find_all(['header', 'nav', 'footer', 'aside', 'script', 'style']):
                tag.decompose()
            
            # Основной контент
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if not main_content:
                main_content = soup.body
            
            if not main_content:
                continue
            
            # Собираем текст с сохранением структуры
            text_parts = []
            for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'td', 'th', 'div']):
                element_text = element.get_text(strip=True, separator=' ')
                if element_text and len(element_text) > 20:  # Увеличили минимальную длину
                    # Добавляем префиксы для заголовков для сохранения структуры
                    if element.name in ['h1', 'h2', 'h3', 'h4']:
                        prefix = "#" * int(element.name[1]) + " "
                        text_parts.append(prefix + element_text)
                    else:
                        text_parts.append(element_text)
            
            if text_parts:
                full_text = "\n".join(text_parts)
                cleaned_text = clean_text(full_text)
                if cleaned_text:
                    docs.append(Document(
                        page_content=cleaned_text,
                        metadata={"source": str(file), "type": "html"}
                    ))
                    logger.info(f"Загружен HTML: {file} - {len(cleaned_text)} символов")
        
        except Exception as e:
            logger.error(f"Ошибка при обработке {file}: {e}")
    
    return docs

def smart_chunking(docs: List[Document]) -> List[Document]:
    """Умное разделение документов с учетом их типа"""
    all_chunks = []
    
    for doc in docs:
        content = doc.page_content
        metadata = doc.metadata.copy()
        
        # Для разных типов документов используем разные стратегии
        doc_type = metadata.get("type", "unknown")
        source = metadata.get("source", "")
        
        if doc_type == "html" or source.endswith(".html"):
            # Для HTML сохраняем структуру по разделам
            sections = []
            current_section = []
            current_title = "Общая информация"
            
            lines = content.split('\n')
            for line in lines:
                line_stripped = line.strip()
                # Определяем заголовки
                if line_stripped.startswith('#'):
                    if current_section:
                        section_text = '\n'.join(current_section)
                        if len(section_text) > 50:
                            sections.append((current_title, section_text))
                        current_section = []
                    # Извлекаем текст заголовка
                    current_title = line_stripped.lstrip('#').strip()
                else:
                    current_section.append(line)
            
            # Добавляем последнюю секцию
            if current_section:
                section_text = '\n'.join(current_section)
                if len(section_text) > 50:
                    sections.append((current_title, section_text))
            
            # Создаем чанки из секций
            for title, section_text in sections:
                # Если секция большая, разбиваем дальше
                if len(section_text) > 1000:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=100,
                        separators=["\n\n", "\n", ". ", "! ", "? "],
                    )
                    sub_chunks = splitter.split_documents([
                        Document(page_content=section_text, metadata=metadata)
                    ])
                    for i, chunk in enumerate(sub_chunks):
                        chunk.metadata.update({
                            **metadata,
                            "section": f"{title} (часть {i+1})"
                        })
                        all_chunks.append(chunk)
                else:
                    chunk = Document(
                        page_content=section_text,
                        metadata={**metadata, "section": title}
                    )
                    all_chunks.append(chunk)
        
        else:
            # Для остальных документов используем стандартный сплиттер
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Увеличил для лучшего контекста
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
                length_function=len,
            )
            chunks = splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata.update(metadata)
            all_chunks.extend(chunks)
    
    logger.info(f"Создано {len(all_chunks)} чанков из {len(docs)} документов")
    return all_chunks

def load_documents() -> List[Document]:
    """Загружает все поддерживаемые документы"""
    all_docs = []
    
    # Загружаем стандартные форматы
    patterns = {
        "**/*.txt": (TextLoader, {"encoding": "utf-8"}),
        "**/*.pdf": (PyPDFLoader, {}),
        "**/*.docx": (Docx2txtLoader, {}),
        "**/*.doc": (Docx2txtLoader, {}),
    }
    
    for glob_pattern, (loader_cls, loader_kwargs) in patterns.items():
        try:
            loader = DirectoryLoader(
                str(DATA_RAW_DIR),
                glob=glob_pattern,
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs,
                show_progress=True,
            )
            docs = loader.load()
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
                # Определяем тип документа по расширению
                source = doc.metadata.get("source", "")
                if source.endswith(".pdf"):
                    doc.metadata["type"] = "pdf"
                elif source.endswith((".docx", ".doc")):
                    doc.metadata["type"] = "doc"
                else:
                    doc.metadata["type"] = "text"
            all_docs.extend(docs)
            logger.info(f"Загружено {len(docs)} документов из {glob_pattern}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить документы {glob_pattern}: {e}")
    
    # Загружаем HTML документы
    html_path = DATA_RAW_DIR / "web"
    if html_path.exists():
        html_docs = load_html_documents(html_path)
        all_docs.extend(html_docs)
        logger.info(f"Загружено {len(html_docs)} HTML документов")
    
    # Удаляем дубликаты и пустые документы
    unique_docs = []
    seen_content = set()
    
    for doc in all_docs:
        if doc.page_content.strip() and len(doc.page_content.strip()) > 50:
            content_hash = hash(doc.page_content[:500])  # Хэш первых 500 символов
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
    
    logger.info(f"После очистки: {len(unique_docs)} уникальных документов")
    return unique_docs

def build_index():
    """Строит векторный индекс документов"""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Начинаем загрузку документов...")
    docs = load_documents()
    
    if not docs:
        logger.warning("Нет документов для индексации")
        print("⚠️ В папке data/raw нет документов. Добавьте файлы в форматах: txt, pdf, docx, html")
        return
    
    logger.info(f"Загружено {len(docs)} документов")
    
    # Умное разделение на чанки
    logger.info("Разделение документов на чанки...")
    chunks = smart_chunking(docs)
    
    if not chunks:
        logger.error("Не удалось создать чанки")
        return
    
    logger.info(f"Создано {len(chunks)} чанков")
    
    # Создаем эмбеддинги
    logger.info("Создание эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Строим индекс
    logger.info("Построение векторного индекса...")
    vectordb = FAISS.from_documents(chunks, embeddings)
    
    # Сохраняем индекс
    vectordb.save_local(str(VECTOR_DB_DIR))
    
    # Дополнительная информация о сохранении
    index_files = list(VECTOR_DB_DIR.glob("*"))
    logger.info(f"Сохранено {len(index_files)} файлов индекса")
    
    print(f"✅ Индекс успешно построен!")
    print(f"📊 Статистика:")
    print(f"   - Документов: {len(docs)}")
    print(f"   - Чанков: {len(chunks)}")
    print(f"   - Средний размер чанка: {sum(len(c.page_content) for c in chunks) // len(chunks)} символов")
    print(f"   - Путь к индексу: {VECTOR_DB_DIR}")

if __name__ == "__main__":
    build_index()
