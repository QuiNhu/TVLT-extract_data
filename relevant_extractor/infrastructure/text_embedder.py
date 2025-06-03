from typing import Dict, Any, List, Text
import torch
from transformers import PhobertTokenizer, RobertaModel

from relevant_extractor.domain.logging_utils import get_logger

class VietnameseLegalEmbedder:
    def __init__(self, model_name: Text = "vinai/phobert-base", device: Text = None):
        self.logger = get_logger(__name__)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = PhobertTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # batch_size x seq_len x hidden_size
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Embed a batch of texts efficiently.
        Returns a tensor of shape (batch_size, embedding_dim).
        """
        with torch.no_grad():
            encoded_input = self.tokenizer.batch_encode_plus(
                texts,
                max_length=218,
                padding='longest',
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            )
            input_ids = encoded_input['input_ids'].to(self.device)
            attention_mask = encoded_input['attention_mask'].to(self.device)

            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.mean_pooling(model_output, attention_mask)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.cpu()

    async def embed_text(self, text: str) -> torch.Tensor:
        """
        Embed a single text by using batch embedding of size 1.
        """
        return self.embed_texts([text.lower()])[0]

    async def embed_laws(self, laws: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Batch embed all laws and attach embedding vectors in-place.
        """
        keys = list(laws.keys())
        texts = [f"{laws[k].get('text', '')}".strip() for k in keys]

        text_embeddings = self.embed_texts(texts)

        for i, k in enumerate(keys):
            laws[k]["embedding"] = text_embeddings[i].tolist()

        return laws