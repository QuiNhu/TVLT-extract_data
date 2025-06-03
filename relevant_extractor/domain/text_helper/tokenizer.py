import re
from re import Pattern
from typing import Text, Dict
import emoji

async def is_emoji(text: Text) -> bool:
    if text in emoji.EMOJI_DATA:
        return True


def get_emoji_regex() -> Pattern:
    """Returns regex to identify emojis."""
    return re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\u200d"  # zero width joiner
        "\u200c"  # zero width non-joiner
        "]+",
        flags=re.UNICODE,
    )

async def remove_emoji(text: Text) -> Text:
    """Remove emoji if the full text matches the emoji regex."""
    text = text.replace(" ", "")
    emoji_pattern = get_emoji_regex()
    match = emoji_pattern.fullmatch(text)

    if match is not None:
        return ""

    return text

def convert_articles_dict_to_text(articles: Dict[str, Dict[str, str]]) -> str:
    """
    Convert dictionary of articles to a structured text format.
    """
    lines = []
    for key, article in articles.items():
        title = article.get("title", "")
        text = article.get("text", "")

        lines.append(f"{key}: {title}")
        lines.append(text.strip())
        lines.append("---")

    return "\n".join(lines)