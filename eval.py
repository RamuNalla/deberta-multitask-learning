import torch
import numpy as np
import evaluate
from tqdm import tqdm
from data import MultiTaskDataBuilder
from model import DebertaMultiTaskModel

def evaluate_mtl_model():
    # 1. Hardware & Metric Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    acc_metric = evaluate.load("accuracy")
    seq_metric = evaluate.load("seqeval")

    # 2. Load the Architecture and the Trained LoRA Weights
    model = DebertaMultiTaskModel().to(device)
    model.encoder.load_adapter("./mtl_lora_adapters", "default")
    model.eval() # Disable dropout and batch normalization

    # 3. Load Validation Data
    # (Assuming MultiTaskDataBuilder has a build_val_dataloaders() method identical 
    # to Day 1, but pulling the "validation" split and setting shuffle=False)
    builder = MultiTaskDataBuilder(batch_size=32)
    val_dataloaders = builder.build_val_dataloaders()

    # 4. Evaluation Loop
    results = {}
    
    with torch.no_grad(): # Disable gradient computation to save memory
        for task_name, dataloader in val_dataloaders.items():
            print(f"Evaluating Task: {task_name}")
            
            all_preds = []
            all_labels = []

            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device) if "labels" in batch else batch["label"].to(device)

                # Forward Pass
                logits = model(input_ids, attention_mask, task_name)
                predictions = torch.argmax(logits, dim=-1)

                # Move tensors to CPU and convert to numpy for metric calculation
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # 5. Task-Specific Metric Computation
            if task_name in ["sentiment", "intent"]:
                task_score = acc_metric.compute(predictions=all_preds, references=all_labels)
                results[task_name] = task_score["accuracy"]
                
            elif task_name == "pos":
                # For sequence evaluation, we must remove the -100 padding/subword tokens
                # and map integer IDs back to their string labels (e.g., 'B-PER', 'O')
                id2label = {i: label for i, label in enumerate(builder.pos_labels)}
                
                true_predictions = [
                    [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(all_preds, all_labels)
                ]
                true_labels = [
                    [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(all_preds, all_labels)
                ]
                
                task_score = seq_metric.compute(predictions=true_predictions, references=true_labels)
                results[task_name] = task_score["overall_f1"]

    print("\n--- Final Evaluation Results ---")
    for task, score in results.items():
        print(f"{task.capitalize()} Score: {score:.4f}")

if __name__ == "__main__":
    evaluate_mtl_model()