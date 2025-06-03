from typing import Text, Dict
import re

async def simple_sent_tokenize(text: Text) -> list:
    """
    Tách câu đơn giản dựa trên dấu câu: ., !, ?
    Giữ lại dấu câu ở cuối câu.
    """
    # Tách câu bằng regex, giữ dấu câu
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]

async def extract_legal_articles(text: Text) -> Dict:
    """
    Extract legal articles from text based on the pattern "Điều <number>."
    Each article contains:
    - 'title': the title of the article (usually the first sentence after "Điều <number>.")
    - 'text': the content of the article (remaining sentences)

    Returns a dictionary structured as:
    {
        "dieu_<number>": {
            "title": <title_string>,
            "text": <content_string>
        },
        ...
    }
    """
    # Regex to find articles starting with "Điều <number>."
    article_pattern = re.compile(r'(Điều\s+(\d+)\.)(.*?)((?=Điều\s+\d+\.)|$)', re.DOTALL | re.IGNORECASE)

    articles = {}

    for match in article_pattern.finditer(text):
        article_number = match.group(2)  # Example: "1"
        article_content = match.group(3).strip()  # Title + body text

        # Extract the article title: use first sentence of the content
        first_line = article_content.split('\n')[0]
        sentences = await simple_sent_tokenize(first_line)
        article_title = sentences[0] if sentences else first_line

        # Extract the remaining text after the title
        remaining_text = article_content[len(article_title):].strip()

        # Split remaining text into sentences for better formatting
        formatted_text = '\n'.join(await simple_sent_tokenize(remaining_text))

        # Store article info in dictionary with key "dieu_<number>"
        articles[f'dieu_{article_number}'] = {
            "title": article_title,
            "text": formatted_text
        }

    return articles