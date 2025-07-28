from mmengine.config import read_base

with read_base():
    from .config_regression_model_stcrowd import *
    from .schedule_stcrowd import *
    from .stcrowd_config import *
    from .runtime import *
