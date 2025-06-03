from relevant_extractor.application.extract_legal_articles_handler import legal_articles_extractor
from relevant_extractor.application.articles_embedder import articles_embedder
from relevant_extractor.application.extract_relevant_articles_hander import get_relevant_articles


__all__ = [
    "legal_articles_extractor",
    "get_relevant_articles",
    "articles_embedder"
]