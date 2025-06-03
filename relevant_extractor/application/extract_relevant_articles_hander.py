from typing import Dict, List, Text
import torch
from collections import Counter

from relevant_extractor.infrastructure import VietnameseLegalEmbedder, rewrite_question
from relevant_extractor.domain import is_emoji, remove_emoji
from relevant_extractor.domain.logging_utils import get_logger


embedder = VietnameseLegalEmbedder()
logger = get_logger(__name__)


async def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    vec1_norm = vec1 / (vec1.norm(p=2) + 1e-9)
    vec2_norm = vec2 / (vec2.norm(p=2) + 1e-9)
    return (vec1_norm * vec2_norm).sum().item()


async def cal_fulltext_score(question: Text, article: Text) -> float:
    question_words = question.lower().split()
    article_words = article.lower().split()

    counter1 = Counter(question_words)
    counter2 = Counter(article_words)

    common_words = set(counter1.keys()) & set(counter2.keys())
    common_count = sum(min(counter1[w], counter2[w]) for w in common_words)
    total_count = len(question_words)

    return round(common_count / total_count, 2) if total_count > 0.0 else 0.0


async def rank_articles_by_similarity(question: str, articles: Dict[str, Dict], top_k: int = 5) -> List:
    """
    Compute cosine similarity between embedded question and articles,
    return sorted list of (article_key, similarity_score) tuples.
    """
    embedded_question = await embedder.embed_text(question)
    embedded_articles = await embedder.embed_laws(articles)

    similarity_scores = []
    for key, article in embedded_articles.items():
        article_embedding = torch.tensor(article["embedding"])
        vector_score = await cosine_similarity(embedded_question.unsqueeze(0), article_embedding.unsqueeze(0))
        fulltext_score = await cal_fulltext_score(question, article["text"])
        if vector_score >= 0.5 and fulltext_score >0.65:
            similarity_scores.append((key, vector_score, fulltext_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Similar Score: {similarity_scores}")
    return similarity_scores[:min(len(similarity_scores)-1, top_k)]


async def get_relevant_articles(question: str, articles: Dict[str, Dict], top_k: int = 5) -> Dict:
    """
    Return top_k relevant articles with their title and text only.
    """
    if await is_emoji(question) or not (await remove_emoji(question)):
        return {
            "message": "User message only contains emoji."
        }

    rewrote_question_response = await rewrite_question(question=question)
    logger.info(f"Rewrite question response: {rewrote_question_response}")
    rewrote_question = rewrote_question_response.get("rewrote_question")
    if not rewrote_question:
        rewrote_question = question

    ranked = await rank_articles_by_similarity(rewrote_question, articles, top_k)
    logger.info(f"Relevant document ranking: {ranked}")
    return {
        key: {
            "title": articles[key]["title"],
            "text": articles[key]["text"]
        } for key, _, _ in ranked
    }
