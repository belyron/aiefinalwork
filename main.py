# main.py

from typing import Dict, List

from config import TOP_K
from data_utils import (
    load_papers,
    get_abstracts,
    print_paper_sample,
    save_results,
)
from keyword_methods import (
    extract_keywords_tfidf,
    extract_keywords_rake,
    extract_keywords_freq_baseline,
    visualize_keyword_scores,
)


def print_menu() -> None:
    print(
        """
===== Keyword Extraction from Research Abstracts =====

1 - Загрузить статьи из JSON
2 - Показать пример статьи
3 - Извлечь ключевые слова TF-IDF
4 - Извлечь ключевые фразы RAKE
5 - Частотный baseline
6 - Сравнительная визуализация (TF-IDF vs baseline)
7 - Обоснование выбора моделей
8 - Сохранить результаты в JSON
9 - Выбрать случайную статью
0 - Выход

Введите номер операции: """,
        end="",
    )


def print_model_justification() -> None:
    """Текстовое обоснование выбора моделей (TF-IDF, RAKE, baseline)."""
    print("\n=== Обоснование выбора моделей ===")
    print(
        """
TF-IDF:
  • Учитывает редкость слова в корпусе.
  • Хорошо выделяет специфические термины научных статей.

RAKE:
  • Извлекает ключевые ФРАЗЫ, а не только отдельные слова.
  • Полезен для выражений вроде "neural network model",
    "natural language processing" и т.п.

Частотный baseline:
  • Очень прост: считает только частоты слов.
  • Служит точкой сравнения: TF-IDF и RAKE должны давать
    более содержательные ключевые слова, чем просто самые частые слова.
"""
    )


def main():
    papers: List[Dict] = []
    results: Dict[str, Dict] = {
        "tfidf": {},
        "rake": {},
        "freq": {},
    }
    current_index = 0

    while True:
        print_menu()
        choice = input().strip()

        # 1 — загрузить статьи
        if choice == "1":
            try:
                papers = load_papers()
                current_index = 0
            except Exception as exc:
                print(f"[!] Данные не были загружены: {exc}")

        # 2 — показать пример статьи
        elif choice == "2":
            if not papers:
                print("[!] Сначала загрузите данные (пункт 1).")
                continue

            try:
                idx_str = input(
                    f"Введите индекс статьи (0..{len(papers)-1}) "
                    f"или Enter для {current_index}: "
                ).strip()
                if idx_str:
                    current_index = int(idx_str)
            except ValueError:
                print("[!] Неверный индекс, используется предыдущий.")

            print_paper_sample(papers, current_index)

        # 3 — TF-IDF
        elif choice == "3":
            if not papers:
                print("[!] Сначала загрузите данные (пункт 1).")
                continue

            abstracts = get_abstracts(papers)
            kw_tfidf = extract_keywords_tfidf(
                abstracts, doc_index=current_index, top_k=TOP_K
            )

            print("\n=== TF-IDF ключевые слова ===")
            for word, score in kw_tfidf:
                print(f"{word:30s} {score:.4f}")

            pid = papers[current_index]["id"]
            results["tfidf"][pid] = {
                "title": papers[current_index]["title"],
                "keywords": [w for w, _ in kw_tfidf],
            }

        # 4 — RAKE
        elif choice == "4":
            if not papers:
                print("[!] Сначала загрузите данные (пункт 1).")
                continue

            text = papers[current_index]["abstract"]
            kw_rake = extract_keywords_rake(text, top_k=TOP_K)

            print("\n=== RAKE ключевые фразы ===")
            # rake_nltk возвращает (score, phrase)
            for score, phrase in kw_rake:
                print(f"{phrase:50s} {score:.4f}")

            pid = papers[current_index]["id"]
            results["rake"][pid] = {
                "title": papers[current_index]["title"],
                "keywords": [phrase for score, phrase in kw_rake],
            }

        # 5 — частотный baseline
        elif choice == "5":
            if not papers:
                print("[!] Сначала загрузите данные (пункт 1).")
                continue

            text = papers[current_index]["abstract"]
            kw_freq = extract_keywords_freq_baseline(text, top_k=TOP_K)

            print("\n=== Частотный baseline ===")
            for word, cnt in kw_freq:
                print(f"{word:30s} {cnt}")

            pid = papers[current_index]["id"]
            results["freq"][pid] = {
                "title": papers[current_index]["title"],
                "keywords": [w for w, _ in kw_freq],
            }

        # 6 — визуализация TF-IDF vs baseline
        elif choice == "6":
            if not papers:
                print("[!] Сначала загрузите данные (пункт 1).")
                continue

            abstracts = get_abstracts(papers)

            tfidf_keywords = extract_keywords_tfidf(
                abstracts, doc_index=current_index, top_k=TOP_K
            )
            freq_keywords = extract_keywords_freq_baseline(
                papers[current_index]["abstract"], top_k=TOP_K
            )

            tfidf_dict = {w: float(s) for w, s in tfidf_keywords}
            freq_dict = {w: float(cnt) for w, cnt in freq_keywords}

            visualize_keyword_scores(tfidf_dict, "TF-IDF scores")
            visualize_keyword_scores(freq_dict, "Frequency baseline")

        # 7 — обоснование выбора моделей
        elif choice == "7":
            print_model_justification()

        # 8 — сохранить результаты
        elif choice == "8":
            save_results(results)

        # 0 — выход
        elif choice == "0":
            print("Выход из программы. Пока!")
            break

        elif choice == "9":
            # выбрать случайную статью
            if not papers:
                print("[!] Сначала загрузите данные (пункт 1).")
                continue

            import random
            current_index = random.randint(0, len(papers) - 1)

            print(f"[OK] Случайная статья выбрана: индекс {current_index}")
            print_paper_sample(papers, current_index)


        else:
            print("[!] Неизвестная команда. Повторите ввод.")


if __name__ == "__main__":
    main()
