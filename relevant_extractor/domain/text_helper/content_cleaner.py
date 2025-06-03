from typing import Text
import re


def convert_money_shorthand(match) -> Text:
    """
    Convert Vietnamese money shorthand notation to full numeric format with 'VNĐ'.

    Supports the following suffixes (case-insensitive):
    - 'k' for thousand (1,000)
    - 'm' for million (1,000,000)
    - 'b' for billion (1,000,000,000)

    Examples:
    - "1.2m" -> "1.200.000 VNĐ"
    - "2k"   -> "2.000 VNĐ"
    - "3b"   -> "3.000.000.000 VNĐ"

    Args:
        match (re.Match): Regex match object containing the numeric part and suffix.

    Returns:
        Text: Converted string with full numeric value and currency suffix.
    """
    # Normalize decimal comma to dot for float conversion
    num_str = match.group(1).replace(',', '.')
    try:
        num = float(num_str)
    except ValueError:
        # Return original text if conversion fails
        return match.group(0)

    suffix = match.group(2).lower()
    multiplier = {'k': 1_000, 'm': 1_000_000, 'b': 1_000_000_000}.get(suffix, 1)
    val = int(num * multiplier)
    # Format number with dots as thousand separators (Vietnamese style)
    val_str = f"{val:,}".replace(',', '.')
    return val_str + " VNĐ"


async def clean_vietnamese_text(text: Text) -> Text:
    """
    Clean Vietnamese text by:
    - Retaining letters, digits, punctuation (. , - : ; ( ) /) and percentage symbol (%)
    - Removing other special characters
    - Normalizing multiple consecutive spaces (2 or more) to a single space
    - Converting money shorthand notation (k, m, b) to full numeric format

    Args:
        text (Text): Raw input text.

    Returns:
        Text: Cleaned and processed text.
    """
    # Compile regex pattern to detect money shorthand (e.g. 1.2m, 3b, 2k)
    money_pattern = re.compile(r'(\d+(?:[.,]\d+)?)([kmbKMB])\b')

    # Remove unwanted characters, keep Vietnamese letters and specified punctuation
    cleaned_text = re.sub(r'[^\w\s\.\,\-\:\;\(\)\/%]', '', text, flags=re.UNICODE)
    # Normalize multiple spaces (2 or more) to single space and trim
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text).strip()

    # Convert shorthand money notations to full numeric format
    cleaned_text = money_pattern.sub(convert_money_shorthand, cleaned_text)

    return cleaned_text


async def remove_footer_conditionally(text: Text, max_words: int = 500) -> Text:
    """
    Conditionally removes the footer section from a legal document.

    The footer is identified by a distinctive phrase starting with:
    "Bộ luật/Luật này (đã) được Quốc hội nước"

    Removal occurs only if the footer (from the match to the end of the text)
    contains less than or equal to `max_words` words.

    Args:
        text (Text): The complete legal document as a string.
        max_words (int): The maximum allowed number of words in the footer
                         to qualify for removal. Defaults to 500.

    Returns:
        Text: The text with the footer removed if conditions are met; otherwise,
             returns the original text unchanged.
    """
    # Compile regex pattern to locate the footer starting phrase up to end of text
    pattern = re.compile(
        r'((?:Luật|Bộ luật) này (?:đã )?được Quốc hội nước.*?\.)(.*)$',
        re.IGNORECASE | re.DOTALL
    )

    match = pattern.search(text)
    if match:
        footer_start = match.start(1)  # Start position of the footer in the text
        footer_text = text[footer_start:]
        footer_word_count = len(footer_text.split())

        # Remove footer only if word count within limit
        if footer_word_count <= max_words:
            return text[:footer_start].rstrip()
        else:
            # Keep full text if footer is too long
            return text
    else:
        # Footer pattern not found; return original text
        return text


# if __name__ == "__main__":
#     number_txt = "Giá trị 1.2M, phí 250.7k, tổng cộng 3b và 1500k."
#     print(clean_vietnamese_text(number_txt))