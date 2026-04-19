
from typing import List, Tuple

import torch
import torch.nn as nn


class CRF(nn.Module):
    """
    Linear-chain CRF.

    Emissions shape: [B, T, C]
    Tags shape:      [B, T]
    Mask shape:      [B, T] (bool)
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags

        # transition scores: from_tag -> to_tag
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def _compute_sequence_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes score for the provided gold tag sequence.
        """
        batch_size, seq_len, _ = emissions.shape

        # start transition + first emission
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, seq_len):
            mask_t = mask[:, t].float()

            transition_score = self.transitions[tags[:, t - 1], tags[:, t]]
            emission_score = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)

            score += (transition_score + emission_score) * mask_t

        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _compute_log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward algorithm in log-space.
        """
        batch_size, seq_len, num_tags = emissions.shape

        alpha = self.start_transitions + emissions[:, 0]  # [B, C]

        for t in range(1, seq_len):
            emission_t = emissions[:, t].unsqueeze(1)      # [B, 1, C]
            transition_scores = self.transitions.unsqueeze(0)  # [1, C, C]
            alpha_expanded = alpha.unsqueeze(2)            # [B, C, 1]

            scores = alpha_expanded + transition_scores + emission_t  # [B, C, C]
            new_alpha = torch.logsumexp(scores, dim=1)     # [B, C]

            mask_t = mask[:, t].unsqueeze(1)
            alpha = torch.where(mask_t, new_alpha, alpha)

        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)  # [B]

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        gold_score = self._compute_sequence_score(emissions, tags, mask)
        log_partition = self._compute_log_partition(emissions, mask)
        return (log_partition - gold_score).mean()

    def viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> List[List[int]]:
        """
        Returns best tag path for each sequence.
        """
        batch_size, seq_len, num_tags = emissions.shape

        score = self.start_transitions + emissions[:, 0]   # [B, C]
        history = []

        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)               # [B, C, 1]
            broadcast_trans = self.transitions.unsqueeze(0)    # [1, C, C]
            next_score = broadcast_score + broadcast_trans     # [B, C, C]

            best_score, best_path = next_score.max(dim=1)      # [B, C], [B, C]
            best_score = best_score + emissions[:, t]          # [B, C]

            mask_t = mask[:, t].unsqueeze(1)
            score = torch.where(mask_t, best_score, score)
            history.append(best_path)

        score = score + self.end_transitions
        best_last_score, best_last_tag = score.max(dim=1)  # [B]

        seq_ends = mask.long().sum(dim=1) - 1
        best_paths = []

        for b in range(batch_size):
            best_tag = best_last_tag[b].item()
            seq_len_b = seq_ends[b].item() + 1

            best_path = [best_tag]
            for hist_t in reversed(history[:seq_len_b - 1]):
                best_tag = hist_t[b][best_tag].item()
                best_path.append(best_tag)

            best_path.reverse()
            best_paths.append(best_path)

        return best_paths