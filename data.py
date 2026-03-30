import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Dict, Iterator, List

class MultiTaskDataBuilder:
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", max_length: int = 128, batch_size: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.batch_size = batch_size

    def _tokenize_and_pad(self, examples: Dict, text_column: str, is_token_classification: bool = False):
        """Strict static padding to avoid PyTorch/XLA graph recompilation."""
        if not is_token_classification:  # for sentiment and intent identification tasks
            return self.tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        else:
            # For POS Tagging, we must align labels with subword tokens. Since Running is Run and ###ning
            tokenized_inputs = self.tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                is_split_into_words=True,
                return_tensors="pt"
            )
            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]       # assign -100 to the second subword and all padded zeroes
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

    def build_dataloaders(self) -> Dict[str, DataLoader]:
        # 1. Sentiment Analysis (SST-2)
        sst2 = load_dataset("glue", "sst2")
        sst2 = sst2.map(lambda x: self._tokenize_and_pad(x, "sentence"), batched=True)
        sst2.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # 2. Intent Classification (Banking77 - High quality ATIS alternative)
        intent = load_dataset("banking77")
        intent = intent.map(lambda x: self._tokenize_and_pad(x, "text"), batched=True)
        intent.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # 3. POS / NER Tagging (CoNLL-2003)
        conll = load_dataset("lhoestq/conll2003")
        conll = conll.map(lambda x: self._tokenize_and_pad(x, "tokens", True), batched=True)
        conll.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        dataloaders = {
            "sentiment": DataLoader(sst2["train"], batch_size=self.batch_size, shuffle=True, drop_last=True),
            "intent": DataLoader(intent["train"], batch_size=self.batch_size, shuffle=True, drop_last=True),
            "pos": DataLoader(conll["train"], batch_size=self.batch_size, shuffle=True, drop_last=True)
        }
        return dataloaders
    
class MultiTaskBatchSampler(Iterator):
    """Yields batches from different tasks in a round-robin fashion."""
    def __init__(self, dataloaders: Dict[str, DataLoader]):
        self.dataloaders = dataloaders
        self.iterators = {task: iter(dl) for task, dl in dataloaders.items()}
        self.task_names = list(dataloaders.keys())
        self.current_task_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        task = self.task_names[self.current_task_idx]
        
        try:
            batch = next(self.iterators[task])
        except StopIteration:
            # If a dataset runs out, reset the iterator for that specific task
            self.iterators[task] = iter(self.dataloaders[task])
            batch = next(self.iterators[task])
            
        # Move to the next task for the next iteration
        self.current_task_idx = (self.current_task_idx + 1) % len(self.task_names)
        
        # Inject the task name into the batch so the model knows which head to route to
        batch["task_name"] = task
        return batch