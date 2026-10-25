"""Group-balanced sampling and the Group DRO training objective."""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from transformers import Trainer


class GroupDROTrainer(Trainer):
    def __init__(self, *args, n_groups=4, group_step_size=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_groups = n_groups
        self.group_step_size = group_step_size
        self.q = torch.ones(n_groups, dtype=torch.float32) / n_groups
        self.model_accepts_loss_kwargs = False
    
    def _get_train_sampler(self, train_dataset=None):
        """Sample groups uniformly in expectation, with replacement."""
        
        ds = self.train_dataset if train_dataset is None else train_dataset
        groups = np.asarray(ds["group_id"], dtype=np.int64)
        counts = np.bincount(groups, minlength=self.n_groups)
        if np.any(counts == 0):
            raise ValueError(f"Empty Group DRO training group: counts={counts.tolist()}")
        per_sample_weight = 1.0 / counts[groups]
        
        return WeightedRandomSampler(
            weights=torch.as_tensor(per_sample_weight, dtype=torch.double),
            num_samples=len(groups),
            replacement=True,
            generator=torch.Generator().manual_seed(self.args.seed),
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        group_id = inputs.pop("group_id", None)
        if group_id is None and model.training:
            raise ValueError("Group DRO training requires group_id in every batch")

        # regular loss for val and test, don't have group ids.
        if group_id is None:
            return super().compute_loss(
                model, inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        example_losses = F.cross_entropy(outputs.logits, labels, reduction="none")

        group_id = group_id.to(device=example_losses.device, dtype=torch.long)
        group_counts = torch.bincount(group_id, minlength=self.n_groups)
        group_sums = torch.zeros(
            self.n_groups,
            device=example_losses.device,
            dtype=example_losses.dtype,
        ).scatter_add_(0, group_id, example_losses)
        group_losses = group_sums / group_counts.clamp_min(1)
        if model.training:
            with torch.no_grad():
                q = self.q.to(example_losses.device)
                # Subtract a common offset to avoid exponential overflow.
                update = self.group_step_size * group_losses.detach().float()
                q = q * torch.exp(update - update.max())
                self.q = (q / q.sum()).detach()
        weighted_loss = (self.q.to(example_losses.device) * group_losses).sum()
        return (weighted_loss, outputs) if return_outputs else weighted_loss

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            checkpoint_dir = os.path.join(
                self._get_output_dir(trial=trial), f"checkpoint-{self.state.global_step}"
            )
            with open(os.path.join(checkpoint_dir, "group_dro_state.json"), "w") as f:
                json.dump({"q": self.q.cpu().tolist()}, f)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        state_path = os.path.join(resume_from_checkpoint, "group_dro_state.json")
        if not os.path.isfile(state_path):
            raise ValueError("Checkpoint is missing Group DRO weights; start a fresh run.")
        with open(state_path) as f:
            q = torch.tensor(json.load(f)["q"], dtype=torch.float32)
        if q.shape != (self.n_groups,) or not torch.isfinite(q).all() or (q < 0).any() or q.sum() <= 0:
            raise ValueError("Invalid Group DRO checkpoint weights")
        self.q = q / q.sum()
