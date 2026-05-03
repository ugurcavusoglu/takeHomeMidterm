import re
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from config import TEXTRANK_NUM_SENTENCES


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def textrank_summarize(text: str, num_sentences: int = TEXTRANK_NUM_SENTENCES) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:num_sentences])

    similarity_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarity_matrix, 0)

    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph, alpha=0.85, max_iter=200)

    ranked_indices = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    selected = sorted(ranked_indices)

    return " ".join(sentences[i] for i in selected)


def run_textrank(articles: list[str], num_sentences: int = TEXTRANK_NUM_SENTENCES) -> list[str]:
    summaries = []
    for article in tqdm(articles, desc="TextRank"):
        summaries.append(textrank_summarize(article, num_sentences))
    return summaries
