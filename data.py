from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Dict, Iterator, List

class MultiTaskDataBuilder:
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", max_length: int = 128, batch_size: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.batch_size = batch_size