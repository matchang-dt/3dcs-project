# /datasets
Datasets go here. They follow a common format. In the supplementary material, MVSplat uses 2 input views and 4 target views for supervision, with the camera baseline increasing as training goes on (some function of # iterations). Though, for evaluation datasets, we should be able to use 2+ views.

If you want to add more datasets (probably not necessary), add your own .py dataset and update `__init__.py`into the Cfg and CfgWrapper abstraction.