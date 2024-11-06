# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from closd.env.tasks import closd_task
from isaacgym.torch_utils import *
from closd.utils.closd_util import STATES

class CLoSDT2M(closd_task.CLoSDTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.init_state = STATES.TEXT2MOTION
        self.hml_data_buf_size = max(self.fake_mdm_args.context_len, self.planning_horizon_20fps)
        self.hml_prefix_from_data = torch.zeros([self.num_envs, 263, 1, self.hml_data_buf_size], dtype=torch.float32, device=self.device)
        return
    
    def update_mdm_conditions(self, env_ids):  
        super().update_mdm_conditions(env_ids)
        
        # updates prompts and lengths
        try:
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        except StopIteration:
            del self.mdm_data_iter
            self.mdm_data_iter = iter(self.mdm_data) # re-initialize
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        for i in env_ids:
            self.hml_prompts[int(i)] = model_kwargs['y']['text'][int(i)]
            self.hml_lengths[int(i)] = model_kwargs['y']['lengths'][int(i)]  
            self.hml_tokens[int(i)] = model_kwargs['y']['tokens'][int(i)]  
            self.db_keys[int(i)] = model_kwargs['y']['db_key'][int(i)]  
        self.hml_prefix_from_data[env_ids] = gt_motion[..., :self.hml_data_buf_size].to(self.device)[env_ids]  # will be used by the first MDM iteration
        if self.cfg['env']['dip']['debug_hml']:
            print(f'in update_mdm_conditions: 1st 10 env_ids={env_ids[:10].cpu().numpy()}, prompts={self.hml_prompts[:2]}')
        return
    
    def get_cur_done(self):
        # Done signal is not in use for this task
        return torch.zeros([self.num_envs], device=self.device, dtype=bool)
    

