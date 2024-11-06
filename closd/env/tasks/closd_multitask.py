# from https://github.com/nv-tlabs/ASE/blob/main/ase/env/tasks/humanoid_strike.py

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

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from closd.env.tasks import closd_task
from closd.utils.rep_util import angle_to_2d_rotmat, is_point_in_rectangle_vectorized

from closd.utils.closd_util import STATES

from closd.diffusion_planner.data_loaders.humanml_utils import HML_JOINT_NAMES
from closd.utils.rep_util import mujoco_2_smpl
import random

class CLoSDMultiTask(closd_task.CLoSDTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # self.init_state_options = [STATES.STRIKE_KICK]
        if cfg['env']['task_filter'] == 'none':
            print('Not applying any task filter.')
            self.init_state_options = [STATES.STRIKE_KICK, STATES.STRIKE_PUNCH, STATES.SIT, STATES.SIT, STATES.REACH, STATES.REACH]
        else:
            print('Applying [{}] task filter.'.format(cfg['env']['task_filter']))
            if cfg['env']['task_filter'] == 'bench':
                self.init_state_options = [STATES.SIT]
            elif cfg['env']['task_filter'] == 'strike':
                self.init_state_options = [STATES.STRIKE_KICK, STATES.STRIKE_PUNCH]
            elif cfg['env']['task_filter'] == 'reach':
                self.init_state_options = [STATES.REACH]
            else:
                raise ValueError()

        self.states_to_eval = list(set(self.init_state_options))
        if STATES.SIT in self.states_to_eval:
            self.states_to_eval += [STATES.GET_UP]

        self.init_state_per_env = random.choices(self.init_state_options, k=cfg.env['num_envs'])
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        # STRIKE - INIT - START
        # self._tar_dist_min = 1.0  # 0.5
        # self._tar_dist_max = 1.5  # 10.0
        # self._near_dist = 1.5
        # self._near_prob = 0.5
        # self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        strike_body_names = cfg["env"]["strikeBodyNames"]
        self._strike_body_ids = self._build_strike_body_ids_tensor(self.envs[0], self.humanoid_handles[0], strike_body_names)
        self._build_target_tensors()
        # self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)  # FIXME - use a real target
        self.box_fall_ang = 0.25
        # STRIKE - INIT - END
        
        self.init_state_per_env = torch.tensor(self.init_state_per_env, dtype=self.cur_state.dtype, device=self.cur_state.device)

        # For is_done calculation
        # self.sit_height = 0.3 # [m]  
        # self.sit_dims = np.array([0.5, 1.8, 0.3])  # TODO - handle mesh asset
        self.sit_dims = np.array([1.2, 2.5, 0.3])  # For mesh mode
        self.sit_safty_margin = 0.1  # for the narrow axis only
        self.max_gap_from_sit = 0.2 # [m]
        # self.contact_force_threshold = 1. # 1.0

        # self.sit_joints = ['Pelvis', 'L_Hip', 'R_Hip', 'L_Hand', 'R_Hand']
        # self.sit_joints = ['Pelvis', 'L_Hip', 'R_Hip']
        # self.sit_joints_idx = [mujoco_joint_names.index(j) for j in self.sit_joints]

        # for bench done calculation
        self.bench_corners = torch.zeros([self.num_envs, 4, 3], dtype=self._rigid_body_pos.dtype, device=self.device)
        self.bench_corners[..., -1] = self.sit_dims[-1]

        self.state_transition_time = 60 # 2 [sec] # number of frames between done was triggered to the state transition
        self.getup_distance = 1.  # FIXME - should be random?
        self.getup_prob = self.cfg['env']['getup_prob']  # After sit  # To disable liedown, use getup_prob=1.

        # Mark the bench left arm relative to the sofa center
        self.perp_arm_delta = 1.2  # [m]
        self.orth_arm_delta = 0.  # [m]
        self.height_arm_delta = 0.  # [m]

        self.target_reset_fn_per_state = {
            STATES.SIT: self._reset_bench_target,
            STATES.REACH: self._reset_reach_target,
            STATES.STRIKE_KICK: self._reset_strike_target,
            STATES.STRIKE_PUNCH: self._reset_strike_target,
            # other states are invalid for task reset
        }

        self.update_machine_fn_per_state = {
            STATES.SIT: self.update_bench_state_machine,
            STATES.REACH: self.update_reach_state_machine,
            STATES.STRIKE_KICK: self.update_strike_state_machine,
            STATES.STRIKE_PUNCH: self.update_strike_state_machine,
            # other states are invalid for task reset
        }

        self.done_fn_per_state = {
            STATES.SIT: self.get_bench_cur_done,
            STATES.REACH: self.get_reach_cur_done,
            STATES.STRIKE_KICK: self.get_strike_cur_done,
            STATES.STRIKE_PUNCH: self.get_strike_cur_done,
            # other states are invalid for task reset
        }


    def _build_strike_body_ids_tensor(self, env_ptr, actor_handle, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids
    
    def get_asset_options(self, task):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0        
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.vhacd_enabled = True            
        asset_options.vhacd_params.max_convex_hulls = 8
        asset_options.vhacd_params.max_num_vertices_per_ch = 128
        asset_options.vhacd_params.resolution = 300000
        
        if task == STATES.STRIKE_KICK:
            asset_options.density = self.cfg['env']['strike_asset_density']
        elif task == STATES.SIT:
            asset_options.density = self.cfg['env']['bench_asset_density']
        elif task == STATES.REACH:
            asset_options.density = 0.
            asset_options.angular_damping = 0.0
            asset_options.linear_damping = 0.0
            asset_options.max_angular_velocity = 0.0
            asset_options.fix_base_link = True
        else:
            raise ValueError()
        
        return asset_options

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._load_target_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return
    
    def _load_target_asset(self):
        asset_root = "closd/data/assets/urdf/"
        
        target_file = "strike_target.urdf"
        target_options = self.get_asset_options(STATES.STRIKE_KICK)
        self._target_asset = self.gym.load_asset(self.sim, asset_root, target_file, target_options)

        bench_file = "sofa.urdf"
        bench_options = self.get_asset_options(STATES.SIT)
        self._bench_asset = self.gym.load_asset(self.sim, asset_root, bench_file, bench_options)

        dummy_file = "location_marker.urdf"
        dummy_options = self.get_asset_options(STATES.REACH)
        self._dummy_asset = self.gym.load_asset(self.sim, asset_root, dummy_file, dummy_options)

        return
    
    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0

        cur_task = self.init_state_per_env[env_id]

        if cur_task in [STATES.STRIKE_PUNCH, STATES.STRIKE_KICK]:
            _asset = self._target_asset
        elif cur_task in [STATES.SIT]:
            _asset = self._bench_asset
        elif cur_task in [STATES.REACH]:
            _asset = self._dummy_asset
        else:
            raise ValueError()
        
        target_handle = self.gym.create_actor(env_ptr, _asset, default_pose, "target", col_group, col_filter, segmentation_id)
        self._target_handles.append(target_handle)
        return

    def get_ids_by_task(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        all_ids = torch.zeros_like(self.is_done)
        all_ids[env_ids] = True
        return {
             _state: (all_ids & (self.init_state_per_env == _state)).nonzero().squeeze(-1) for _state in STATES
        }
    
    def _reset_target(self, env_ids):
        ids_by_task = self.get_ids_by_task(env_ids)
        for _state in STATES:
            if len(ids_by_task[_state]) > 0:
                self.target_reset_fn_per_state[_state](ids_by_task[_state])
        return
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        # self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return
    
    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_target(env_ids)
        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))   
        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)
        return


    def update_state_machine(self):
        if not self.cfg.env['enable_transitions']:
            return
        ids_by_task = self.get_ids_by_task()
        for _state in STATES:
            if len(ids_by_task[_state]) > 0:
                self.update_machine_fn_per_state[_state](ids_by_task[_state])
        return
        

    def update_strike_state_machine(self, env_ids):
        # If done sitting -> get up
        task_ids = torch.zeros_like(self.is_done)
        task_ids[env_ids] = True

        # done_was_n_frames_ago = self.progress_buf - self.last_done
        is_strike = torch.logical_or(self.cur_state == STATES.STRIKE_KICK, self.cur_state == STATES.STRIKE_PUNCH)
        done_strike = self.is_done & is_strike &  task_ids
        switch_to_halt = done_strike
        all_switch = switch_to_halt
        self.cur_state[switch_to_halt] = STATES.HALT
        self.is_done[all_switch] = False  # starting a new task
        # print(done_was_n_frames_ago, self.cur_state)
        
        if torch.any(all_switch):
            self.update_mdm_conditions(all_switch.nonzero())

        if torch.any(switch_to_halt):
            halt_pose = torch.zeros_like(self._tar_pos[switch_to_halt])
            halt_pose[:, :2] = self._rigid_body_pos[switch_to_halt, mujoco_2_smpl[HML_JOINT_NAMES.index('pelvis')], :2]
            halt_pose[:, 2] = 0.93  # standing height 0.93
            self._tar_pos[switch_to_halt] = halt_pose
        return
    
    def update_bench_state_machine(self, env_ids):
        # If done sitting -> get up
        task_ids = torch.zeros_like(self.is_done)
        task_ids[env_ids] = True

        done_was_n_frames_ago = self.progress_buf - self.last_done
        done_sit = torch.logical_and(done_was_n_frames_ago == self.state_transition_time, self.cur_state == STATES.SIT)  
        _rand = torch.bernoulli(self.getup_prob * torch.ones([self.num_envs], device=done_sit.device, dtype=done_sit.dtype)).to(bool)
        switch_to_getup = done_sit & _rand & task_ids
        all_switch = switch_to_getup
        self.cur_state[switch_to_getup] = STATES.GET_UP
        self.is_done[all_switch] = False  # starting a new task
        # print(done_was_n_frames_ago, self.cur_state)
        
        if torch.any(all_switch):
            self.update_mdm_conditions(all_switch.nonzero())

        if torch.any(switch_to_getup):
            getup_deltas = torch.zeros_like(self._tar_pos[switch_to_getup])
            getup_deltas[:, 0] = self.getup_distance * torch.sin(self._target_rot_theta[switch_to_getup])
            getup_deltas[:, 1] = -self.getup_distance * torch.cos(self._target_rot_theta[switch_to_getup])
            getup_deltas[:, 2] = 0.53  # standing height 0.93
            self._tar_pos[switch_to_getup] += getup_deltas
        
        return

    def update_reach_state_machine(self, env_ids):
        # TODO - implement
        return

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        self._target_rotmat = torch.zeros([self.num_envs, 2, 2], dtype=self._target_states.dtype, device=self._target_states.device)
        self._target_rot_theta = torch.zeros([self.num_envs], dtype=self._target_states.dtype, device=self._target_states.device)
        # self._tar_pos =  self._target_states[..., 0:3].clone()  # Guy - for goal reaching
        # self._aux_tar_pos =  torch.zeros_like(self._tar_pos)  # Marks the place for the head when lying down
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1 #  (num_actors-1)  #  1 + 1 # 25
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

        return
    
    def set_random_target_loc(self, env_ids, task):
        n = len(env_ids)
        _tar_dist_min, _tar_dist_max = self.state_machine_conditions[task]['tar_dist_range']
        rand_dist = (_tar_dist_max - _tar_dist_min) * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device) + _tar_dist_min
        
        rand_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        self._target_states[env_ids, 0] = rand_dist * torch.cos(rand_theta) + self._humanoid_root_states[env_ids, 0]
        self._target_states[env_ids, 1] = rand_dist * torch.sin(rand_theta) + self._humanoid_root_states[env_ids, 1]
        self._target_states[env_ids, 2] = 0.

        rand_rot_theta = rand_theta - np.pi # [-pi, pi]
        rand_rot_theta += np.pi/2. 
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_states.dtype, device=self._target_states.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis)

        self._target_rot_theta[env_ids] = rand_rot_theta
        self._target_rotmat[env_ids] = angle_to_2d_rotmat(self._target_rot_theta[env_ids])
        self._target_states[env_ids, 3:7] = rand_rot
        self._target_states[env_ids, 7:10] = 0.0
        self._target_states[env_ids, 10:13] = 0.0
        return
    
    def _reset_reach_target(self, env_ids):
        self.set_random_target_loc(env_ids, STATES.REACH)
        self._tar_pos[env_ids, :] =  self._target_states[env_ids, 0:3].clone()
        return
    
    def _reset_strike_target(self, env_ids):
        self.set_random_target_loc(env_ids, STATES.STRIKE_KICK)
        self._tar_pos[env_ids, :] =  self._target_states[env_ids, 0:3].clone()
        self._tar_pos[env_ids, -1] += 0.9  # TODO - conditioned on state (kick/punch)
        return
    
    
    def _reset_bench_target(self, env_ids):
        self.set_random_target_loc(env_ids, STATES.SIT)

        self._tar_pos[env_ids, :] =  self._target_states[env_ids, 0:3].clone()  # Guy - for goal reaching
        self._tar_pos[env_ids, -1] += 0.4  # bench height
      
        self._aux_tar_pos[env_ids] = self._tar_pos[env_ids].clone()  # Marks the place for the head when lying down
        self._aux_tar_pos[env_ids, 0] += self.orth_arm_delta * torch.sin(self._target_rot_theta[env_ids]) + self.perp_arm_delta * torch.cos(self._target_rot_theta[env_ids])
        self._aux_tar_pos[env_ids, 1] += -self.orth_arm_delta * torch.cos(self._target_rot_theta[env_ids]) + self.perp_arm_delta * torch.sin(self._target_rot_theta[env_ids])
        self._aux_tar_pos[env_ids, 2] += self.height_arm_delta

        vec_perp = torch.cat([self.sit_dims[1] / 2. * torch.cos(self._target_rot_theta[env_ids])[:, None], self.sit_dims[1] / 2. * torch.sin(self._target_rot_theta[env_ids])[:, None]], dim=-1)
        vec_orth = torch.cat([self.sit_dims[0] / 2. * torch.sin(self._target_rot_theta[env_ids])[:, None], -self.sit_dims[0] / 2. * torch.cos(self._target_rot_theta[env_ids])[:, None]], dim=-1)
        self.bench_corners[env_ids, 0, :2] = self._tar_pos[env_ids, :2] + vec_perp + vec_orth
        self.bench_corners[env_ids, 1, :2] = self._tar_pos[env_ids, :2] - vec_perp + vec_orth
        self.bench_corners[env_ids, 2, :2] = self._tar_pos[env_ids, :2] - vec_perp - vec_orth
        self.bench_corners[env_ids, 3, :2] = self._tar_pos[env_ids, :2] + vec_perp - vec_orth

        return


    def get_cur_done(self):
        ids_by_task = self.get_ids_by_task()
        all_dones = torch.zeros_like(self.is_done)
        for _state in STATES:
            if len(ids_by_task[_state]) > 0:
                _dones = self.done_fn_per_state[_state](ids_by_task[_state])
                all_dones |= _dones
        return all_dones


    def get_bench_cur_done(self, env_ids):
        task_ids = torch.zeros_like(self.is_done)
        task_ids[env_ids] = True

        in_bench_box = is_point_in_rectangle_vectorized(points=self._rigid_body_pos[:, 0, :2], rect_corners=self.bench_corners[..., :2])
        pelvis_height = self._rigid_body_pos[:, 0, 2]
        in_bench_height = torch.logical_and(self.sit_dims[2] <= pelvis_height, pelvis_height <= self.sit_dims[2] + self.max_gap_from_sit)
        done_sitting = in_bench_box & in_bench_height


        # cur_loc = self._rigid_body_pos[:, self.target_idx[0], self.target_coordinates[0]]
        cur_loc = self._rigid_body_pos[:, mujoco_2_smpl[HML_JOINT_NAMES.index('pelvis')]]
        target_loc = self._tar_pos  # [:, self.target_coordinates[0]]
        done_getup = torch.linalg.vector_norm(target_loc-cur_loc, dim=-1) <= self.done_dist

        done = torch.where(self.cur_state == STATES.SIT, done_sitting, done_getup)
        done = ((self.cur_state == STATES.SIT) & done_sitting) | ((self.cur_state == STATES.GET_UP) & done_getup)
        done = done & task_ids
        return done

    def get_strike_cur_done(self, env_ids):
        task_ids = torch.zeros_like(self.is_done)
        task_ids[env_ids] = True

        # Done if the box is on the ground
        tar_rot = self._target_states[:, 3:7]
        up = torch.zeros_like(tar_rot[:, :-1])
        up[..., -1] = 1
        tar_up = quat_rotate(tar_rot, up)
        tar_rot_err = torch.sum(up * tar_up, dim=-1)
        done_strike = tar_rot_err <= self.box_fall_ang

        cur_loc = self._rigid_body_pos[:, mujoco_2_smpl[HML_JOINT_NAMES.index('pelvis')], :2]
        target_loc = self._tar_pos[:, :2]  # [:, self.target_coordinates[0]]
        done_halt = torch.linalg.vector_norm(target_loc-cur_loc, dim=-1) <= self.done_dist

        is_strike = torch.logical_or(self.cur_state == STATES.STRIKE_KICK, self.cur_state == STATES.STRIKE_PUNCH)
        done = (is_strike & done_strike) | ((self.cur_state == STATES.HALT) & done_halt)
        done = done & task_ids
        return done
    
    def get_reach_cur_done(self, env_ids):
        task_ids = torch.zeros_like(self.is_done)
        task_ids[env_ids] = True

        # calc if the task was done in the current timestep for each env
        if self.multi_target_cond:
            cur_loc = self._rigid_body_pos[torch.arange(self.num_envs), self.mujoco_joint_idx[:, 0]].clone()
            cur_loc[self.is_2d_target[:, 0].to(bool), -1] = 0
            target_loc = self._tar_pos.clone()
            target_loc[self.is_2d_target[:, 0].to(bool), -1] = 0
        else:
            cur_loc = self._rigid_body_pos[:, self.target_idx[0], self.target_coordinates[0]]
            target_loc = self._tar_pos[:, self.target_coordinates[0]]
        done = torch.linalg.vector_norm(target_loc-cur_loc, dim=-1) <= self.done_dist
        return done.clone() & task_ids