from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import os
import torch

class ViewSet:
    def __init__(self,
        images: torch.Tensor | None,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ):
        # these should be all index aligned and in sorted temporal order with 'timestamps'
        self.images = images # (V, H, W, 3)
        self.intrinsics = intrinsics # (V, 3, 3)
        self.extrinsics = extrinsics # (V, 4, 4)

    def _debug_save(self, output_path: str = "default_view_debug_directory"):
        # DEBUG: save all view data into an output directory
        for i in range(self.images.shape[0]):
            image = self.images[i].cpu().numpy()
            intrinsics = self.intrinsics[i].cpu().numpy()
            extrinsics = self.extrinsics[i].cpu().numpy()
            torch.save(image, os.path.join(output_path, f"image_{i}.pt"))
            torch.save(intrinsics, os.path.join(output_path, f"intrinsics_{i}.pt"))
            torch.save(extrinsics, os.path.join(output_path, f"extrinsics_{i}.pt"))

class ViewSampler(ABC):
    """
    Samples views into input and target ViewSets for training.
    """
    def __init__(self, cfg, stage: str): # cfg is a subtree of DatasetCfg
        self.cfg = cfg
        self.stage = stage  # 'train', 'val', 'test' (change as part of dataset filepaths)
    
    @abstractmethod
    def sample_views(self,
        all_views: ViewSet,
        **kwargs
    ) -> tuple[ViewSet, ViewSet]: # input & target ViewSets
        # from a full ViewSample, sample context and target ViewSets
        pass

# one used in MVSplat paper where camera baselines gradually expand during training
# if 'test' or 'val' stage, then we just use uniform sampling instead
class ViewSamplerDefault(ViewSampler):
    def sample_views(self,
        all_views: ViewSet,
        **kwargs
    ) -> tuple[ViewSet, ViewSet]: # input & target ViewSets
        num_total_views = all_views.extrinsics.shape[0]
        num_input_views = self.cfg.num_input_views
        num_target_views = self.cfg.num_target_views
        curr_iters = kwargs.get('curr_train_step', None)
        max_iters = kwargs.get('max_train_steps', 300000)

        if num_input_views < 2:
            raise ValueError("Expects num_input_views >= 2 for now")

        # Use a fraction for spread: at the start, close to 0; at the end, nearly 1
        spread_frac = min(1.0, max(0.0, curr_iters / max_iters))

        # Exponential or root ramp for smoother transition
        ramp = spread_frac ** 0.5

        # The maximum offset from the anchor the input pair can be (num_total_views-2)
        max_offset = int((num_total_views-2) * ramp)
        if max_offset < 1:
            max_offset = 1
            
        # To avoid out-of-bounds, the anchor can be placed in [0, num_total_views - max_offset - 1]
        anchor_range = num_total_views - max_offset
        if anchor_range <= 0:
            anchor_range = 1

        anchor_idx = torch.randint(0, anchor_range, (1,)).item()
        # To keep the pair close at the start: offset is sampled from [1, max_offset]
        offset = torch.randint(1, max_offset+1, (1,)).item()
        pair_idx = anchor_idx + offset

        # select indices for context (input) and target views
        input_indices = [anchor_idx, pair_idx]
        # If more than 2 input views are needed, interpolate or random sample between:
        if num_input_views > 2:
            additional = torch.randperm(pair_idx - anchor_idx - 1)[:num_input_views-2] + (anchor_idx+1) if pair_idx - anchor_idx - 1 > 0 else torch.tensor([])
            input_indices = [anchor_idx] + additional.tolist() + [pair_idx]
            input_indices = sorted(input_indices)

        # Now, for target views: pick from the set not in input_indices
        all_indices = set(range(num_total_views))
        input_set = set(input_indices)
        remaining_indices = sorted(list(all_indices - input_set))
        if len(remaining_indices) < num_target_views:
            # If not enough, just roll over the first ones
            target_indices = remaining_indices + input_indices[:(num_target_views-len(remaining_indices))]
        else:
            target_indices = torch.randperm(len(remaining_indices))[:num_target_views]
            target_indices = [remaining_indices[i] for i in target_indices]

        # Gather the involved views
        def subset_view(viewset, indices):
            images = viewset.images[indices] if viewset.images is not None else None
            intrinsics = viewset.intrinsics[indices]
            extrinsics = viewset.extrinsics[indices]
            return ViewSet(images, intrinsics, extrinsics)

        context_views = subset_view(all_views, input_indices)
        target_views = subset_view(all_views, target_indices)

        return context_views, target_views

        



