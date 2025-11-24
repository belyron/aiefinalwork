# keyword_methods.py

from collections import Counter
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import nltk
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer

from config import TOP_K

# --- подготовка NLTK ресурсов ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def extract_keywords_tfidf(
    docs: List[str],
    doc_index: int = 0,
    top_k: int = TOP_K
) -> List[Tuple[str, float]]:
    """
    Извлекает ключевые слова из документа (по индексу) методом TF-IDF.
    Возвращает список (слово, score).
    """
    if not docs:
        print("[!] Нет документов для TF-IDF.")
        return []

    if doc_index < 0 or doc_index >= len(docs):
        print("[!] Неверный индекс документа.")
        return []

    vectorizer = TfidfVectorizer(
        max_df=0.8,
        min_df=1,
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    row = tfidf_matrix[doc_index].toarray().ravel()
    top_idx = row.argsort()[::-1][:top_k]

    keywords = [(feature_names[i], float(row[i])) for i in top_idx if row[i] > 0]
    return keywords


def extract_keywords_rake(
    text: str,
    top_k: int = TOP_K
) -> List[Tuple[str, float]]:
    """
    Извлекает ключевые фразы методом RAKE.
    Возвращает список (фраза, score).
    """
    if not text.strip():
        return []

    r = Rake(language="english")
    r.extract_keywords_from_text(text)
    ranked_phrases_with_scores = r.get_ranked_phrases_with_scores()
    return ranked_phrases_with_scores[:top_k]


def extract_keywords_freq_baseline(
    text: str,
    top_k: int = TOP_K
) -> List[Tuple[str, int]]:
    """
    Простой baseline:
    считаем частоты слов (без стоп-слов и пунктуации).
    Возвращает (слово, частота).
    """
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import string

    if not text.strip():
        return []

    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())

    table = str.maketrans("", "", string.punctuation)
    cleaned_tokens = [
        w.translate(table)
        for w in tokens
        if w.translate(table) and w.translate(table) not in stop_words
    ]

    counter = Counter(cleaned_tokens)
    return counter.most_common(top_k)


def visualize_keyword_scores(scores: Dict[str, float], title: str) -> None:
    """
    Визуализация словарей {слово: значение} с помощью matplotlib.
    Строит столбчатую диаграмму и показывает её в отдельном окне.
    Дополнительно можно сохранить график в PNG.
    """
    if not scores:
        print("[!] Нет данных для визуализации.")
        return

    words = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(10, 5))
    plt.bar(words, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()

    # необязательно, но полезно для отчёта: сохранить картинку
    safe_title = title.lower().replace(" ", "_")
    filename = f"{safe_title}.png"
    plt.savefig(filename, dpi=150)

    print(f"[OK] График '{title}' сохранён в файл: {filename}")

    # показать окно с графиком
    plt.show()
