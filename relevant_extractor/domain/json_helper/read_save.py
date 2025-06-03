from typing import Optional, Dict, Any, Text
import os
import json

from relevant_extractor.domain.logging_utils import get_logger

logger = get_logger(__name__)

async def read_json_file(file_path: Text) -> Optional[Dict[Text, Any]]:
    if not isinstance(file_path, str):
        logger.error("Error: File path must be a string.")
        return None

    if not os.path.exists(file_path):
        logger.error(f"Error: File does not exist: <{file_path}>")
        return None

    if not os.path.isfile(file_path):
        logger.error(f"Error: Path is not a file: <{file_path}>")
        return None

    if not file_path.lower().endswith('.json'):
        logger.error("Error: File extension is not '.json'.")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        logger.info(f"LOADED <{file_path}>")
        data = json.load(f)
    return data


async def save_json_file(data: Dict, file_path: Text) -> bool:
    if not isinstance(file_path, str):
        logger.error("Error: File path must be a string.")
        return False

    if not file_path.lower().endswith('.json'):
        logger.error("Error: File extension must be '.json'.")
        return False

    # Check if directory exists, create if not
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        logger.error(f"Error: Directory does not exist: <{dir_path}>")
        return False

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"SAVED <{dir_path}>")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file: {e}")
        return False


async def process_json_files_in_dataset(folder_path: Text="/Users/cerris/PycharmProjects/TVPL-test/dataset", end: Text= ".json"):
    json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(end)]

    return json_files


