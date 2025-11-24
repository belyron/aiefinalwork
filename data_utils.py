# data_utils.py

import json
from typing import List, Dict

from config import DATA_PATH, RESULTS_PATH, N_DOCS


def iter_json_records(path: str):
    """
    Генератор JSON-записей.
    Поддерживает:
      • JSON-массив: [ {...}, {...}, ... ]
      • JSONL:       {...}\n{...}\n
    """
    with open(path, "r", encoding="utf-8") as f:
        # читаем первый значащий символ, чтобы понять формат
        first_char = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
                break

        # JSONL
        if first_char and first_char != "[":
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
            return

        # JSON-массив
        f.seek(0)
        data = json.load(f)
        for item in data:
            yield item


def load_papers(path: str = DATA_PATH,
                n_docs: int = N_DOCS) -> List[Dict]:
    """
    Загружает до n_docs статей и оставляет только:
    id, title, abstract, authors.
    """
    papers: List[Dict] = []
    print(f"[INFO] Читаем JSON из {path} ...")

    for record in iter_json_records(path):
        pid = record.get("id")
        title = record.get("title")
        abstract = record.get("abstract")
        authors = record.get("authors")

        if not (pid and title and abstract and authors):
            continue

        papers.append({
            "id": str(pid),
            "title": str(title).strip(),
            "abstract": str(abstract).strip(),
            "authors": str(authors).strip(),
        })

        if len(papers) >= n_docs:
            break

    if not papers:
        raise ValueError("Не удалось загрузить ни одной корректной записи.")

    print(f"[OK] Загружено статей: {len(papers)}")
    return papers


def get_abstracts(papers: List[Dict]) -> List[str]:
    """Возвращает список только аннотаций (abstract)."""
    return [p["abstract"] for p in papers]


def save_results(results: Dict, path: str = RESULTS_PATH) -> None:
    """Сохранение результатов (ключевых слов) в JSON."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[OK] Результаты сохранены в {path}")
    except Exception as exc:
        print(f"[ERROR] Не удалось сохранить файл: {exc}")


def print_paper_sample(papers: List[Dict], index: int = 0) -> None:
    """Печатает одну статью по индексу."""
    if not papers:
        print("[!] Сначала загрузите данные (пункт меню 1).")
        return

    if index < 0 or index >= len(papers):
        print("[!] Индекс вне диапазона.")
        return

    paper = papers[index]
    print(f"\n=== Статья #{index} ===")
    print(f"ID      : {paper['id']}")
    print(f"Title   : {paper['title']}")
    print(f"Authors : {paper['authors']}")
    print("-" * 70)
    print(paper["abstract"])
    print("=" * 70)
