
import torch
from msamp import clip_grad_norm_


class NativeScalerWithGradReduce:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, model, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if hasattr(optimizer, 'all_reduce_grads'):
            optimizer.all_reduce_grads(model)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            assert clip_mode == 'norm'
            clip_grad_norm_(parameters, clip_grad)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)