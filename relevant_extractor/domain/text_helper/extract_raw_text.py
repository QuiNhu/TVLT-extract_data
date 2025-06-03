import re
from typing import Text


async def is_line_to_remove(line: Text) -> bool:
    """
    Determine whether a line should be removed from the text based on specific criteria:
    - The line is empty or contains only whitespace.
    - The line matches the pattern for section headers starting with "Chương" followed by Roman or Arabic numerals.
    - The line matches the pattern for subsections starting with "Mục" followed by a number.
    - The line is predominantly uppercase (>= 80% of letters are uppercase), which often indicates titles or headers.

    Args:
        line (Text): A single line of text.

    Returns:
        bool: True if the line should be removed; False otherwise.
    """
    line_strip = line.strip()
    if not line_strip:
        return True

    # Match lines like "Chương II" or "Chương 3"
    if re.match(r'^Chương\s+[\w\dIVXLCDM]+\.?$', line_strip, re.IGNORECASE):
        return True

    # Match lines like "Mục 1." or "Mục 2 Some title"
    if re.match(r'^Mục\s+\d+\.?.*$', line_strip, re.IGNORECASE):
        return True

    # Remove lines where at least 80% of letters are uppercase (likely headers)
    letters = [c for c in line_strip if c.isalpha()]
    if letters:
        uppercase_count = sum(1 for c in letters if c.isupper())
        if uppercase_count / len(letters) >= 0.8:
            return True

    return False


async def read_txt(txt_path: Text) -> Text:
    """
    Read the text file, filter out unwanted lines based on specific rules,
    and return the cleaned text as a single string.

    Filtering criteria are implemented in the is_line_to_remove function.

    Args:
        txt_path (str): The file path to the text file.

    Returns:
        Text: The cleaned text after filtering and joining the remaining lines with newlines.
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    filtered_lines = [line.strip() for line in lines if not await is_line_to_remove(line)]

    full_text = "\n".join(filtered_lines)

    return full_text



