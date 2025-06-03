from typing import Text, Dict

from relevant_extractor.domain import (
    read_txt,
    remove_footer_conditionally,
    clean_vietnamese_text,
    extract_legal_articles
)


async def legal_articles_extractor(file_path: Text) -> Dict:
    """
    Extract legal articles from a given Vietnamese text file.

    This function reads the content of a text file specified by `file_path`,
    then processes the text by cleaning Vietnamese-specific characters,
    conditionally removing footers, and finally extracting legal articles
    structured as a dictionary.

    Args:
        file_path (Text): The file path to the input .txt file containing legal text.

    Returns:
        Dict: A dictionary where keys are article identifiers and values
              contain the extracted article content including title and text.
    """
    text = await read_txt(txt_path=file_path)

    text = await clean_vietnamese_text(text=text)
    text = await remove_footer_conditionally(text=text)

    articles = await extract_legal_articles(text=text)
    return articles


