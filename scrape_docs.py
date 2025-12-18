"""
Скрипт для парсинга документов с веб-сайтов (например, сайта Сбера).

Использование:
    # Парсинг текста со страницы:
    python scrape_docs.py <URL> [--output <имя_файла.txt>]
    
    # Скачивание документов (PDF, DOCX и т.д.) со страницы:
    python scrape_docs.py <URL> --download-docs [--limit <количество>]
    
Примеры:
    python scrape_docs.py https://www.sberbank.ru/ru/person/help/info/faq
    python scrape_docs.py https://www.sberbank.ru/... --output faq_sber.txt
    python scrape_docs.py https://www.sberbank.ru/... --download-docs
    python scrape_docs.py https://www.sberbank.ru/... --download-docs --limit 10
"""

import argparse
import re
from pathlib import Path
from typing import List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from rag_config import DATA_RAW_DIR


def clean_text(text: str) -> str:
    """Очищает текст от лишних пробелов и переносов."""
    # Убираем множественные пробелы и переносы
    text = re.sub(r'\s+', ' ', text)
    # Убираем пробелы в начале и конце строк
    text = re.sub(r'\n\s+', '\n', text)
    return text.strip()


def extract_text_from_html(html: str, url: str) -> str:
    """
    Извлекает основной текст из HTML-страницы.
    Удаляет навигацию, футеры, скрипты и стили.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Удаляем ненужные элементы
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        tag.decompose()
    
    # Ищем основной контент (обычно в <main>, <article>, или <div class="content">)
    main_content = (
        soup.find('main') or
        soup.find('article') or
        soup.find('div', class_=re.compile(r'content|main|text', re.I)) or
        soup.find('body')
    )
    
    if main_content:
        text = main_content.get_text(separator='\n', strip=True)
    else:
        # Если не нашли основной контент, берём весь body
        text = soup.get_text(separator='\n', strip=True)
    
    # Добавляем URL в начало для отслеживания источника
    text = f"Источник: {url}\n\n{text}"
    
    return clean_text(text)


def scrape_url(url: str, output_file: str = None) -> Path:
    """
    Парсит страницу по URL и сохраняет текст в файл.
    
    Args:
        url: URL страницы для парсинга
        output_file: Имя выходного файла (если не указано, генерируется из URL)
    
    Returns:
        Path к сохранённому файлу
    """
    # Создаём папку для данных, если её нет
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Генерируем имя файла из URL, если не указано
    if not output_file:
        parsed_url = urlparse(url)
        # Берём последнюю часть пути или домен
        path_parts = [p for p in parsed_url.path.split('/') if p]
        if path_parts:
            output_file = path_parts[-1] + '.txt'
        else:
            output_file = parsed_url.netloc.replace('.', '_') + '.txt'
    
    # Убираем недопустимые символы из имени файла
    output_file = re.sub(r'[<>:"/\\|?*]', '_', output_file)
    if not output_file.endswith('.txt'):
        output_file += '.txt'
    
    output_path = DATA_RAW_DIR / output_file
    
    print(f"Загружаем страницу: {url}")
    
    # Заголовки, чтобы сайт думал, что это обычный браузер
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or 'utf-8'
        
        print(f"Извлекаем текст из HTML...")
        text = extract_text_from_html(response.text, url)
        
        # Сохраняем в файл
        output_path.write_text(text, encoding='utf-8')
        print(f"✓ Текст сохранён в: {output_path}")
        print(f"  Размер: {len(text)} символов")
        
        return output_path
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Ошибка при загрузке страницы: {e}")
        raise
    except Exception as e:
        print(f"✗ Ошибка при обработке: {e}")
        raise


def find_document_links(url: str, debug: bool = False) -> List[tuple]:
    """
    Находит все ссылки на документы (PDF, DOC, DOCX, XLS, XLSX, TXT) на странице.
    Ищет как прямые ссылки на файлы, так и ссылки на страницы с документами.
    
    Args:
        url: URL страницы для парсинга
        debug: Показывать отладочную информацию
    
    Returns:
        Список кортежей (url_документа, имя_файла, тип_документа)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    }
    
    # Расширения документов, которые мы ищем
    doc_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt', '.rtf']
    
    # Ключевые слова в тексте ссылок, которые могут указывать на документы
    doc_keywords = ['скачать', 'download', 'pdf', 'doc', 'документ', 'файл', 'file']
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        html_content = response.text
        
        # Сохраняем HTML для отладки, если включен debug
        if debug:
            debug_file = DATA_RAW_DIR / 'debug_page.html'
            debug_file.write_text(html_content, encoding='utf-8')
            print(f"  HTML страницы сохранён в: {debug_file}")
            print(f"  Размер HTML: {len(html_content)} символов")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        document_links = []
        all_links = []
        
        # Ищем все ссылки <a href="...">
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            link_text = link.get_text(strip=True).lower()
            
            all_links.append((full_url, link_text))
            
            # Проверяем, является ли ссылка документом по расширению
            parsed = urlparse(full_url)
            path_lower = parsed.path.lower()
            query_lower = parsed.query.lower()
            
            found_ext = None
            for ext in doc_extensions:
                if ext in path_lower or ext in query_lower:
                    found_ext = ext
                    break
            
            # Также проверяем по тексту ссылки или атрибутам
            is_doc_link = False
            if not found_ext:
                # Проверяем текст ссылки на ключевые слова
                for keyword in doc_keywords:
                    if keyword in link_text:
                        # Проверяем, может быть это ссылка на страницу с документом
                        # Попробуем проверить Content-Type по HEAD-запросу
                        try:
                            head_response = requests.head(full_url, headers=headers, timeout=10, allow_redirects=True)
                            content_type = head_response.headers.get('Content-Type', '').lower()
                            if any(ext.replace('.', '') in content_type for ext in doc_extensions):
                                # Определяем расширение по Content-Type
                                if 'pdf' in content_type:
                                    found_ext = '.pdf'
                                elif 'msword' in content_type or 'word' in content_type:
                                    found_ext = '.docx'
                                elif 'excel' in content_type or 'spreadsheet' in content_type:
                                    found_ext = '.xlsx'
                                is_doc_link = True
                                break
                        except:
                            pass
                        
                        # Если не удалось определить по Content-Type, но есть ключевое слово
                        # и ссылка выглядит как документ (нет расширения .html, .php и т.д.)
                        if not is_doc_link and not any(x in path_lower for x in ['.html', '.php', '.aspx', '/']):
                            # Предполагаем PDF по умолчанию для таких ссылок
                            found_ext = '.pdf'
                            is_doc_link = True
                            break
            
            if found_ext:
                # Пытаемся получить имя файла из ссылки или текста ссылки
                filename = parsed.path.split('/')[-1] or link_text or f"document_{len(document_links) + 1}"
                
                # Если имя файла не содержит расширение, добавляем его
                if not filename.endswith(found_ext):
                    filename += found_ext
                
                # Убираем недопустимые символы
                filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                
                # Убираем лишние пробелы и делаем имя файла безопасным
                filename = re.sub(r'\s+', '_', filename)
                if len(filename) > 200:
                    filename = filename[:200] + found_ext
                
                document_links.append((full_url, filename, found_ext))
        
        if debug:
            print(f"  Всего ссылок <a> на странице: {len(all_links)}")
            print(f"  Найдено ссылок с расширениями документов: {len(document_links)}")
            
            # Также ищем другие возможные элементы со ссылками
            all_elements_with_href = soup.find_all(attrs={'href': True})
            print(f"  Всего элементов с href: {len(all_elements_with_href)}")
            
            # Ищем возможные ссылки в data-атрибутах или JavaScript
            script_tags = soup.find_all('script')
            if script_tags:
                print(f"  Найдено <script> тегов: {len(script_tags)}")
                # Ищем URL в скриптах
                script_urls = []
                for script in script_tags:
                    script_text = script.string or ''
                    # Ищем паттерны типа "url": "...", href: "...", и т.д.
                    urls_in_script = re.findall(r'["\']([^"\']*\.(?:pdf|doc|docx|xls|xlsx))["\']', script_text, re.I)
                    script_urls.extend(urls_in_script)
                if script_urls:
                    print(f"  Найдено URL документов в JavaScript: {len(script_urls)}")
                    for i, script_url in enumerate(script_urls[:5], 1):
                        print(f"    {i}. {script_url[:100]}")
            
            if len(document_links) == 0 and len(all_links) == 0:
                print("  ⚠️  Ссылки не найдены. Возможные причины:")
                print("     - Страница загружается через JavaScript (нужен Selenium)")
                print("     - Ссылки находятся в iframe или другом формате")
                print("     - Проверьте сохранённый HTML файл для анализа")
            elif len(document_links) == 0:
                print("  Примеры найденных ссылок (первые 10):")
                for i, (link_url, link_text) in enumerate(all_links[:10], 1):
                    print(f"    {i}. {link_url[:80]}... (текст: {link_text[:50]})")
        
        return document_links
        
    except Exception as e:
        print(f"✗ Ошибка при поиске документов: {e}")
        return []


def download_document(doc_url: str, filename: str, output_dir: Path = None) -> Path:
    """
    Скачивает документ по URL и сохраняет его.
    
    Args:
        doc_url: URL документа
        filename: Имя файла для сохранения
        output_dir: Папка для сохранения (по умолчанию DATA_RAW_DIR)
    
    Returns:
        Path к сохранённому файлу
    """
    if output_dir is None:
        output_dir = DATA_RAW_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    
    try:
        print(f"  Скачиваем: {filename}")
        response = requests.get(doc_url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()
        
        # Сохраняем файл
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = output_path.stat().st_size
        print(f"    ✓ Сохранён: {output_path} ({file_size:,} байт)")
        return output_path
        
    except Exception as e:
        print(f"    ✗ Ошибка при скачивании {filename}: {e}")
        raise


def download_documents_from_page(url: str, limit: int = None, debug: bool = False) -> List[Path]:
    """
    Находит и скачивает документы со страницы.
    
    Args:
        url: URL страницы для парсинга
        limit: Максимальное количество документов для скачивания (None = все)
        debug: Показывать отладочную информацию
    
    Returns:
        Список путей к скачанным файлам
    """
    print(f"Ищем документы на странице: {url}")
    document_links = find_document_links(url, debug=debug)
    
    if not document_links:
        print("  Документы не найдены.")
        return []
    
    print(f"  Найдено документов: {len(document_links)}")
    
    # Ограничиваем количество, если указан limit
    if limit is not None and limit > 0:
        document_links = document_links[:limit]
        print(f"  Будет скачано (ограничение): {len(document_links)}")
    
    downloaded_files = []
    
    for doc_url, filename, doc_type in document_links:
        print(f"\n  [{len(downloaded_files) + 1}/{len(document_links)}] {doc_type.upper()}: {filename}")
        try:
            file_path = download_document(doc_url, filename)
            downloaded_files.append(file_path)
        except Exception as e:
            print(f"    Пропускаем из-за ошибки: {e}")
            continue
    
    print(f"\n✓ Всего скачано: {len(downloaded_files)} документов")
    return downloaded_files


def scrape_multiple_urls(urls: List[str], output_prefix: str = None) -> List[Path]:
    """
    Парсит несколько URL и сохраняет каждый в отдельный файл.
    
    Args:
        urls: Список URL для парсинга
        output_prefix: Префикс для имён файлов (опционально)
    
    Returns:
        Список путей к сохранённым файлам
    """
    saved_files = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Обрабатываем: {url}")
        
        output_name = None
        if output_prefix:
            output_name = f"{output_prefix}_{i:02d}.txt"
        
        try:
            file_path = scrape_url(url, output_name)
            saved_files.append(file_path)
        except Exception as e:
            print(f"Пропускаем {url} из-за ошибки: {e}")
            continue
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description='Парсинг документов с веб-сайтов для RAG-индексации'
    )
    parser.add_argument(
        'urls',
        nargs='+',
        help='URL страниц для парсинга (можно указать несколько)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Имя выходного файла (для одного URL)'
    )
    parser.add_argument(
        '--prefix', '-p',
        help='Префикс для имён файлов (для нескольких URL)'
    )
    parser.add_argument(
        '--download-docs', '-d',
        action='store_true',
        help='Скачать документы (PDF, DOCX и т.д.) со страницы вместо парсинга текста'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Максимальное количество документов для скачивания (по умолчанию - все)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Показать отладочную информацию (все найденные ссылки)'
    )
    
    args = parser.parse_args()
    
    if args.download_docs:
        # Режим скачивания документов
        if len(args.urls) == 1:
            download_documents_from_page(args.urls[0], limit=args.limit, debug=args.debug)
        else:
            print("В режиме --download-docs можно указать только один URL")
    else:
        # Обычный режим парсинга текста
        if len(args.urls) == 1:
            scrape_url(args.urls[0], args.output)
        else:
            saved = scrape_multiple_urls(args.urls, args.prefix)
            print(f"\n✓ Всего обработано: {len(saved)} файлов")


if __name__ == '__main__':
    main()

