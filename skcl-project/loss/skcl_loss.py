"""
skcl_loss.py

Implements Eq. (4), the semantic knowledge-driven contrastive loss:

    L_SKCL(x_i) = -( (1/|M_i|) * sum_{c_q in M_i} log[ exp(z_i . c_q / tau') /
                                                         sum_{j=1}^n exp(z_i . c_j / tau') ]
                    + (1/|P_i|) * sum_{z_p in P_i} log[ exp(z_i . z_p / tau) /
                                                         sum_{l in Y} (1/|A_l|) sum_{z_k in A_l} exp(z_i . z_k / tau) ] )

Where (paper's notation):
  - M_i = {y_i} union TopK-neighbors-of-y_i  (prototype indices, spans the
    FULL graph -- fine class y_i's neighbors may be fine OR coarse nodes)
  - c_q = prototype vector for graph node q
  - n = total number of prototypes (all graph nodes, fine+coarse)
  - P_i = other samples in the batch sharing y_i's fine label (excludes i)
  - A_l = samples in the batch belonging to fine class l
  - tau' > tau > 0 (paper: tau=0.1, tau'=0.2 for CIFAR)

The second term structurally matches BCL's class-balanced contrastive
denominator (same normalize-per-class-then-sum-over-classes structure),
so it's implemented the same way as BalSCL in contrastive.py, extended
with the new prototype-alignment first term.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class SKCLLoss(nn.Module):
    def __init__(self, tau: float = 0.1, tau_prime: float = 0.2):
        super().__init__()
        if not (tau_prime > tau > 0):
            raise ValueError(f"Paper requires tau' > tau > 0, got tau={tau}, tau'={tau_prime}")
        self.tau = tau
        self.tau_prime = tau_prime

    def forward(
        self,
        z_bar: torch.Tensor,          # (batch, feat_dim) L2-normalized sample representations
        targets: torch.Tensor,        # (batch,) fine-class labels
        prototypes: torch.Tensor,     # (n_nodes, feat_dim) L2-normalized, fine-then-coarse order
        neighbor_lists: List[List[int]],  # neighbor_lists[fine_class] = list of Top-K neighbor node indices
    ) -> torch.Tensor:
        device = z_bar.device
        batch_size = z_bar.shape[0]

        # ---- Term 1: prototype alignment (pull toward M_i = {y_i} U TopK(y_i)) ----
        proto_logits = (z_bar @ prototypes.T) / self.tau_prime  # (batch, n_nodes)
        proto_log_denom = torch.logsumexp(proto_logits, dim=1, keepdim=True)
        proto_log_prob = proto_logits - proto_log_denom

        proto_term = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            y_i = int(targets[i].item())
            m_i = [y_i] + list(neighbor_lists[y_i])
            proto_term[i] = proto_log_prob[i, m_i].mean()

        # ---- Term 2: class-balanced contrastive term (same structure as BCL's BalSCL) ----
        targets_col = targets.contiguous().view(-1, 1)
        mask = torch.eq(targets_col, targets_col.T).float().to(device)
        self_mask = torch.eye(batch_size, device=device)
        pos_mask = mask - self_mask

        sim = (z_bar @ z_bar.T) / self.tau
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        exp_sim = torch.exp(sim)

        classes_in_batch = torch.unique(targets)
        class_avg = torch.zeros(batch_size, batch_size, device=device)
        for l in classes_in_batch:
            l_mask = (targets == l).float().view(1, -1)
            l_count = l_mask.sum()
            if l_count > 0:
                class_avg += (exp_sim * l_mask) / l_count
        denom = class_avg.sum(dim=1, keepdim=True)
        denom = denom.clamp_min(1e-12)

        log_prob = sim - torch.log(denom)

        pos_count = pos_mask.sum(dim=1)
        contrast_term = torch.zeros(batch_size, device=device)
        has_positives = pos_count > 0
        if has_positives.any():
            contrast_term[has_positives] = (
                (pos_mask[has_positives] * log_prob[has_positives]).sum(dim=1) / pos_count[has_positives]
            )

        loss_per_sample = -(proto_term + contrast_term)
        return loss_per_sample.mean()


if __name__ == "__main__":
    # CPU sanity check with synthetic data.
    torch.manual_seed(0)
    batch_size, feat_dim, n_fine, n_coarse = 16, 32, 10, 2
    n_nodes = n_fine + n_coarse

    z_bar = torch.nn.functional.normalize(torch.randn(batch_size, feat_dim), dim=1)
    targets = torch.randint(0, n_fine, (batch_size,))
    prototypes = torch.nn.functional.normalize(torch.randn(n_nodes, feat_dim), dim=1)

    # Fake Top-2 neighbor lists per fine class (arbitrary but valid indices into n_nodes)
    neighbor_lists = [[(i + 1) % n_nodes, (i + 2) % n_nodes] for i in range(n_fine)]

    criterion = SKCLLoss(tau=0.1, tau_prime=0.2)
    loss = criterion(z_bar, targets, prototypes, neighbor_lists)
    print(f"Loss value: {loss.item():.4f}")
    assert not torch.isnan(loss), "loss is NaN"
    assert loss.item() > 0, "loss should be positive (negative log-likelihood)"

    # Gradient check
    leaf = torch.randn(batch_size, feat_dim, requires_grad=True)
    z_bar_grad_test = torch.nn.functional.normalize(leaf, dim=1)
    loss2 = criterion(z_bar_grad_test, targets, prototypes, neighbor_lists)
    loss2.backward()
    print(f"Gradient reached input: {leaf.grad is not None and leaf.grad.abs().sum().item() > 0}")
    print("OK: SKCLLoss forward+backward run cleanly on synthetic data")
