from typing import Text, Dict

from relevant_extractor.domain import read_json_file, save_json_file, process_json_files_in_dataset
from relevant_extractor.infrastructure import VietnameseLegalEmbedder

embedder = VietnameseLegalEmbedder()

async def articles_embedder(folder_path: Text=None, content: Dict=None, file_path: Text=None) -> bool:

    if folder_path:
        file_paths = await process_json_files_in_dataset(folder_path=folder_path)
        for path in file_paths:
            content = await read_json_file(file_path=path)
            if not content:
                return False
            embedded_content = embedder.embed_laws(laws=content)
            return await save_json_file(data=embedded_content, file_path=path)

    if content:
        embedded_content = embedder.embed_laws(laws=content)
        return await save_json_file(data=embedded_content, file_path="NoName.json")

    if file_path:
        content = await read_json_file(file_path=file_path)
        if not content:
            return False
        embedded_content = embedder.embed_laws(laws=content)
        return await save_json_file(data=embedded_content, file_path=file_path)

