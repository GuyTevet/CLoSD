import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from closd.diffusion_planner.utils.misc import wrapped_getattr
import joblib

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, guidance_type='text'):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        self.guidance_type = guidance_type


    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        if 'text' in self.guidance_type:
            y_uncond['text_uncond'] = True
        if 'target' in self.guidance_type:
            y_uncond['target_uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))

    def __getattr__(self, name, default=None):
        # this method is reached only if name is not in self.__dict__.
        return wrapped_getattr(self, name, default=None)


class AutoRegressiveSampler():
    def __init__(self, args, sample_fn, required_frames=196):
        self.sample_fn = sample_fn
        self.args = args
        self.required_frames = required_frames
    
    def sample(self, model, shape, **kargs):
        bs = shape[0]
        n_iterations = (self.required_frames + self.args.pred_len - 1) // self.args.pred_len
        samples_buf = []
        cur_prefix = deepcopy(kargs['model_kwargs']['y']['prefix'])  # init with data
        if self.args.autoregressive_include_prefix:
            samples_buf.append(cur_prefix)
        autoregressive_shape = list(deepcopy(shape))
        autoregressive_shape[-1] = self.args.pred_len
        for _ in range(n_iterations):
            cur_kargs = deepcopy(kargs)
            cur_kargs['model_kwargs']['y']['prefix'] = cur_prefix
            sample = self.sample_fn(model, autoregressive_shape, **cur_kargs)
            samples_buf.append(sample.clone()[..., -self.args.pred_len:])
            cur_prefix = sample.clone()[..., -self.args.context_len:]  # update

        full_batch = torch.cat(samples_buf, dim=-1)[..., :self.required_frames]  # 200 -> 196
        return full_batch