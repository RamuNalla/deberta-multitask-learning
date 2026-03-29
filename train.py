import torch
import torch.nn as nn
import wandb
import torch_xla.core.xla_model as xm
from tqdm import tqdm

from data import MultiTaskDataBuilder, MultiTaskBatchSampler
from model import DebertaMultiTaskModel
from loss import UncertaintyLoss

def train_mtl_model(epochs=3, lr=2e-4):
    # 1. Hardware & Environment Setup
    device = xm.xla_device()
    print(f"Running on device: {device}")
    
    wandb.init(project="deberta-multitask-learning", name="tpu-lora-run-1")

    # 2. Initialize Components
    builder = MultiTaskDataBuilder(batch_size=32)
    dataloaders = builder.build_dataloaders()
    train_sampler = MultiTaskBatchSampler(dataloaders)
    
    model = DebertaMultiTaskModel().to(device)
    loss_balancer = UncertaintyLoss(num_tasks=3).to(device)
    
    # 3. Optimizer setup (Combining Model LoRA weights + Loss Sigma weights)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': lr},
        {'params': loss_balancer.parameters(), 'lr': lr * 10} # Loss params learn faster
    ])

    # Task-specific raw loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_token = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. The Training Loop
    model.train()
    loss_balancer.train()
    
    total_steps = len(dataloaders["sentiment"]) + len(dataloaders["intent"]) + len(dataloaders["pos"])
    
    for epoch in range(epochs):
        progress_bar = tqdm(range(total_steps), desc=f"Epoch {epoch+1}")
        
        for step in progress_bar:
            optimizer.zero_grad()
            
            # Step A: Pull batch and move to TPU
            batch = next(train_sampler)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device) if "labels" in batch else batch["label"].to(device)
            task_name = batch["task_name"]

            # Step B: Forward Pass
            logits = model(input_ids, attention_mask, task_name)

            # Step C: Calculate Raw Loss
            if task_name in ["sentiment", "intent"]:
                raw_loss = criterion_cls(logits, labels)
            else: # POS tagging requires flattening the tensors
                raw_loss = criterion_token(logits.view(-1, logits.shape[-1]), labels.view(-1))

            # Step D: Apply Dynamic Weighting
            # We map task names to specific indices in our UncertaintyLoss parameter array
            task_idx = {"sentiment": 0, "intent": 1, "pos": 2}[task_name]
            
            # Create a list of 0s, inject our active loss, and pass to balancer
            losses = [torch.tensor(0.0, device=device)] * 3
            losses[task_idx] = raw_loss
            weighted_loss = loss_balancer(losses)

            # Step E: Backward Pass & TPU Optimization
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            xm.optimizer_step(optimizer)

            # Step F: MLOps Logging
            if step % 50 == 0:
                wandb.log({
                    f"raw_loss_{task_name}": raw_loss.item(),
                    "total_weighted_loss": weighted_loss.item(),
                    "sigma_sentiment": torch.exp(loss_balancer.log_vars[0]).item(),
                    "sigma_intent": torch.exp(loss_balancer.log_vars[1]).item(),
                    "sigma_pos": torch.exp(loss_balancer.log_vars[2]).item(),
                })
                progress_bar.set_postfix({"loss": weighted_loss.item(), "task": task_name})

    # Save final LoRA adapters and loss weights
    model.encoder.save_pretrained("./mtl_lora_adapters")
    torch.save(loss_balancer.state_dict(), "./mtl_loss_weights.pt")
    wandb.finish()

if __name__ == "__main__":
    train_mtl_model()