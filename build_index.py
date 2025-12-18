import re
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


def clean_text(text: str) -> str:
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

    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def load_html_documents(path: Path) -> List[Document]:
    """Загружает HTML-файлы из папки и превращает их в документы для индекса"""
    docs = []
    for file in path.glob("*.html"):
        try:
            with open(file, encoding="utf-8") as f:
                html = f.read()
            soup = BeautifulSoup(html, "html.parser")
            text = ' '.join(soup.stripped_strings)
            if text:
                docs.append(Document(page_content=text, metadata={"source": str(file)}))
        except Exception as e:
            print(f"Ошибка при обработке {file}: {e}")
    return docs


def load_documents() -> List[Document]:
    """Загружает все поддерживаемые документы + HTML"""
    all_docs = []

    # 1️⃣ Старые форматы
    patterns = {
        "**/*.txt": (TextLoader, {"encoding": "utf-8"}),
        "**/*.pdf": (PyPDFLoader, {}),
        "**/*.docx": (Docx2txtLoader, {}),
        "**/*.doc": (Docx2txtLoader, {}),
    }

    for glob, (cls, kwargs) in patterns.items():
        loader = DirectoryLoader(
            str(DATA_RAW_DIR),
            glob=glob,
            loader_cls=cls,
            loader_kwargs=kwargs,
        )
        docs = loader.load()
        for d in docs:
            d.page_content = clean_text(d.page_content)
            all_docs.append(d)

    # 2️⃣ HTML страницы
    html_path = DATA_RAW_DIR / "web"
    if html_path.exists():
        html_docs = load_html_documents(html_path)
        all_docs.extend(html_docs)

    return all_docs


def build_index():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_documents()
    if not docs:
        print("Нет документов для индексации")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(str(VECTOR_DB_DIR))

    print(f"✅ Индекс построен: {len(chunks)} чанков")


if __name__ == "__main__":
    build_index()
