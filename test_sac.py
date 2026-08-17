import torch
log_alpha = torch.tensor(0.0, requires_grad=True)
log_probs = torch.tensor([-1.0])
target_entropy = -2.0
alpha_loss = -(log_alpha * (log_probs + target_entropy)).mean()
alpha_loss.backward()
print("Gradient of log_alpha:", log_alpha.grad.item())
