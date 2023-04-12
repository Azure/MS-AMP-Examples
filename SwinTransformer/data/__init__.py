
from .build import build_loader as _build_loader

def build_loader(config, simmim=False, is_pretrain=False):
    assert not simmim
    assert not is_pretrain
    if not simmim:
        return _build_loader(config)