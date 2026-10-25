"""Dynamic padding while keeping group IDs out of model inputs."""
import torch
from transformers import DataCollatorWithPadding


class GroupDROCollator:
    def __init__(self, tokenizer):
        self.pad = DataCollatorWithPadding(tokenizer=tokenizer)
        
    def __call__(self, features):
        groups = [f["group_id"] for f in features] if "group_id" in features[0] else None
        
        token_features = [
            {k: f[k] for k in ("input_ids", "attention_mask", "label") if k in f}
            for f in features
        ]
        batch = self.pad(token_features)
        if groups is not None:
            batch["group_id"] = torch.tensor(groups, dtype=torch.long)
        return batch
