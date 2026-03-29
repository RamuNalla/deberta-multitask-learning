import torch
import torch.nn as nn

class UncertaintyLoss(nn.Module):
    """
    Implements Homoscedastic Task Uncertainty Weighting.
    Automatically balances the gradients of multiple tasks during joint training.
    """
    def __init__(self, num_tasks: int = 3):
        super(UncertaintyLoss, self).__init__()
        # Initialize log variances at 0 (meaning variance = 1)
        # We use log variance for numerical stability to avoid division by zero
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        if len(losses) != len(self.log_vars):
            raise ValueError("Number of losses must match number of tasks.")

        total_loss = 0
        for i, loss in enumerate(losses):
            # L_i * exp(-log_var) + log_var
            precision = torch.exp(-self.log_vars[i])
            total_loss += (precision * loss) + self.log_vars[i]

        return total_loss