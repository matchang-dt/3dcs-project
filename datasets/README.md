# /datasets
Datasets go here. They follow a common format. In the supplementary material, MVSplat uses 2 input views and 4 target views for supervision, with the camera baseline increasing as training goes on (some function of # iterations). Though, for evaluation, we should be able to use 2+ views for input.

If you want to add more datasets (probably not necessary), add your own .py dataset and update `__init__.py`.

## batch contents
All datasets have the following output interface per batch:

```
batch = {
    'context': {
        'images': context_views.images,          # [num_input_views, 3, H, W]
        'intrinsics': context_views.intrinsics,  # [num_input_views, 3, 3]
        'extrinsics': context_views.extrinsics,  # [num_input_views, 4, 4]
    },
    'target': {
        'images': target_views.images,          # [num_target_views, 3, H, W]
        'intrinsics': target_views.intrinsics,  # [num_target_views, 3, 3]
        'extrinsics': target_views.extrinsics,  # [num_target_views, 4, 4]
    },
    'scene_key': scene_dir.name,
    'near_plane': 0.1,  # Default values, can be overridden by subclasses
    'far_plane': 100.0,
}
```

Intrinsics are normalized and resolution-independent.
Extrinsics are pre-centered and normalized based on the farthest two cameras of the FULL set of cameras per scene.
Extrinsics are in W2C OpenCV camera coordinate convention (camera frame: x(right), y(down), z(inward)).
The world frame seems to place objects at -z up, with the XY plane as the 'floor' plane for these scenes.

The batch only contains sampled (not total!) input and target images whose number are described by the user in the hydra config for the experiment. 
In the beginning of training, these are sampled closely together with a small camera baseline between input views. However, as training goes on,
larger baselines will have larger sampling probability after a point.

Near and far planes are also given (also a necessary component), and may need to be tuned differently to different datasets. 