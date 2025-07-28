# ------------------------------------------------------------------------------------------------
# Copyright (c) MonoDETR
# ------------------------------------------------------------------------------------------------
from three_d.model import build_monodetr


def build_model(cfg):
    return build_monodetr(cfg)
