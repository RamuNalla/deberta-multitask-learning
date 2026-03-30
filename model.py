import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

class DebertaMultiTaskModel(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", 
                 num_intents: int = 77, num_pos_tags: int = 9):
        super(DebertaMultiTaskModel, self).__init__()
        
        # 1. Load the shared foundational backbone
        base_model = AutoModel.from_pretrained(model_name)
        
        # 2. Inject LoRA adapters to make training parameter-efficient
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=16, 
            target_modules=["query_proj", "value_proj"], 
            lora_dropout=0.1,
            bias="none"
        )
        self.encoder = get_peft_model(base_model, lora_config)
        
        hidden_size = base_model.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # 3. Task-Specific Prediction Heads
        # Sentiment (Binary Classification: Positive/Negative)
        self.sentiment_head = nn.Linear(hidden_size, 2)
        
        # Intent Classification (e.g., 77 classes for Banking77)
        self.intent_head = nn.Linear(hidden_size, num_intents)
        
        # POS Tagging (Token Classification: 9 classes for CoNLL-2003)
        self.pos_head = nn.Linear(hidden_size, num_pos_tags)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, task_name: str):
        # Pass data through the shared LoRA-adapted encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state 
        
        # --- THE FIX: Align the dtypes ---
        # Cast the 16-bit backbone output to 32-bit to match our custom Linear heads
        sequence_output = sequence_output.to(torch.float32)
        
        sequence_output = self.dropout(sequence_output)

        # Route the tensor to the correct head based on the task
        if task_name == "sentiment":
            # Use the [CLS] token equivalent (Index 0) for sentence-level tasks
            cls_token_state = sequence_output[:, 0, :] 
            logits = self.sentiment_head(cls_token_state)
            
        elif task_name == "intent":
            # Use the [CLS] token equivalent (Index 0)
            cls_token_state = sequence_output[:, 0, :]
            logits = self.intent_head(cls_token_state)
            
        elif task_name == "pos":
            # Use the entire sequence of tokens for word-level tagging
            logits = self.pos_head(sequence_output)
            
        else:
            raise ValueError(f"Unrecognized task_name: {task_name}")

        return logits