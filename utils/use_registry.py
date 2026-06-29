from .registry import Registry

MODELS = Registry('model')
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
MEMORIES = Registry('memory')
LOSSES = Registry('loss')
