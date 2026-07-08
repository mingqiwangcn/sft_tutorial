"""
Dynamic padding collator for SFT.
"""

from dataclasses import dataclass

import torch


@dataclass
class SFTDataCollator:
    pad_token_id: int
    label_pad_token_id: int = -100
    pad_to_multiple_of: int | None = 8

    def _get_padded_length(self, features: list[dict]) -> int:
        max_length = max(len(feature["input_ids"]) for feature in features)

        if self.pad_to_multiple_of is None:
            return max_length

        remainder = max_length % self.pad_to_multiple_of
        if remainder == 0:
            return max_length

        return max_length + self.pad_to_multiple_of - remainder

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        padded_length = self._get_padded_length(features)

        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            seq_length = len(feature["input_ids"])
            pad_length = padded_length - seq_length

            input_ids.append(
                feature["input_ids"] + [self.pad_token_id] * pad_length
            )
            attention_mask.append(
                feature["attention_mask"] + [0] * pad_length
            )
            labels.append(
                feature["labels"] + [self.label_pad_token_id] * pad_length
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
