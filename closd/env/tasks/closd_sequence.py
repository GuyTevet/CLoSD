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
import numpy as np

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from closd.env.tasks import closd_multitask
from closd.env.tasks.closd_task import CLoSDTask
from closd.utils.rep_util import angle_to_2d_rotmat, is_point_in_rectangle_vectorized

from closd.utils.closd_util import STATES

from closd.diffusion_planner.data_loaders.humanml_utils import HML_JOINT_NAMES
from closd.utils.rep_util import mujoco_2_smpl
import random

class CLoSDSequence(closd_multitask.CLoSDMultiTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.actor_id_offsets = [STATES.REACH, STATES.STRIKE_PUNCH, STATES.SIT]
        self.state_machine_sequence = [STATES[_state_str] for _state_str in cfg['env']['state_machine']]
        self.init_state_options = [self.state_machine_sequence[0]]
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.demo_script = [STATES.SIT, STATES.GET_UP, STATES.REACH, STATES.STRIKE_KICK]
        self.state_to_tar_idx = {
            STATES.SIT: 2, STATES.GET_UP: 2, STATES.STRIKE_PUNCH: 1, STATES.STRIKE_KICK: 1, STATES.HALT: 1, STATES.REACH: 0
        }

        self.next_state = {}
        for _state in STATES:
            self.next_state[_state] = _state
            try: 
                _next_seq_idx = self.state_machine_sequence.index(_state) + 1
                if _next_seq_idx < len(self.state_machine_sequence):
                    self.next_state[_state] = self.state_machine_sequence[_next_seq_idx]
            except ValueError:
                pass
        
        if cfg['env']['reach_dist_range'] is not None:
            self.state_machine_conditions[STATES.REACH]['tar_dist_range'] = cfg['env']['reach_dist_range']
        if cfg['env']['bench_dist_range'] is not None:
            self.state_machine_conditions[STATES.SIT]['tar_dist_range'] = cfg['env']['bench_dist_range']
        if cfg['env']['strike_dist_range'] is not None:
            self.state_machine_conditions[STATES.STRIKE_KICK]['tar_dist_range'] = cfg['env']['strike_dist_range']
            self.state_machine_conditions[STATES.STRIKE_PUNCH]['tar_dist_range'] = cfg['env']['strike_dist_range']


    def _reset_target(self, env_ids):
        self._reset_bench_target(env_ids)
        self._reset_strike_target(env_ids)
        self._reset_reach_target(env_ids)
        return
    
    
    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0

        # cur_task = self.init_state_per_env[env_id]
        is_bed = False

        for cur_task in self.actor_id_offsets:
            if cur_task in [STATES.STRIKE_PUNCH, STATES.STRIKE_KICK]:
                _asset = self._target_asset
            elif cur_task in [STATES.SIT]:
                _asset = self._bed_asset if is_bed else self._bench_asset
            elif cur_task in [STATES.REACH]:
                _asset = self._dummy_asset
            else:
                raise ValueError()
            
            target_handle = self.gym.create_actor(env_ptr, _asset, default_pose, "target", col_group, col_filter, segmentation_id)
            self._target_handles.append(target_handle)
        return
    
    
    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1:, :]
        # self._target_rotmat = torch.zeros([self.num_envs, 2, 2], dtype=self._target_states.dtype, device=self._target_states.device)
        self._target_rot_theta = torch.zeros([self.num_envs, 3], dtype=self._target_states.dtype, device=self._target_states.device)
        self._tar_pos =  self._target_states[..., 0:3].clone()  # Guy - for goal reaching
        self._aux_tar_pos =  torch.zeros_like(self._tar_pos)  # Marks the place for the head when lying down
        self._tar_actor_ids = [to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + i + 1 for i in range(3)] #  (num_actors-1)  #  1 + 1 # 25
        
        # bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        # contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        # contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        # self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

        return
    
    def _reset_env_tensors(self, env_ids):
        # super()._reset_env_tensors(env_ids)
        super(closd_multitask.CLoSDMultiTask, self)._reset_env_tensors(env_ids)

        env_ids_int32 = torch.cat(self._tar_actor_ids)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))  

        return

    # def calc_cur_target_multi_joint(self, max_dist=1.2):
    def calc_cur_target_multi_joint(self, max_dist=1.8):
        # FIXME - currently supporting first tartget only
        # returns target [n_envs, 3] cuted for max_dist [m] from current location
        all_start_loc = []
        all_end_loc = []
        all_tars = [self._tar_pos, self._aux_tar_pos]
        tar_idx = torch.tensor([self.state_to_tar_idx[int(s)] for s in self.cur_state], device=self.device, dtype=torch.int64)
        for joint_i in range(2):
            start_loc = self._rigid_body_pos[torch.arange(self.num_envs), self.mujoco_joint_idx[:, joint_i]]
            end_loc = all_tars[joint_i][torch.arange(self.num_envs), tar_idx].clone()
            end_loc -= start_loc
            traj_length = torch.linalg.norm(end_loc, dim=-1)[:, None]
            traj_length_2d = torch.linalg.norm(end_loc[..., :2], dim=-1)[:, None]  # for traj
            traj_length = torch.cat([traj_length, traj_length_2d], dim=1)[torch.arange(self.num_envs), self.is_2d_target[:, joint_i]][:, None]
            needs_shortening = (traj_length > max_dist).float()
            shortening_factor = (traj_length*needs_shortening/max_dist) + (1.*(1. - needs_shortening))
            end_loc /= shortening_factor
            end_loc += start_loc

            # print('debug_ends', self.debug_ends)
            all_start_loc.append(start_loc[:, None])
            all_end_loc.append(end_loc[:, None]) 

        all_start_loc = torch.cat(all_start_loc, dim=1)
        all_end_loc = torch.cat(all_end_loc, dim=1)

        # update marker
        self.debug_starts = all_start_loc.clone()
        self.debug_ends = all_end_loc.clone()

        return all_end_loc
    
    def set_random_target_loc(self, env_ids, task):
        n = len(env_ids)
        _tar_dist_min, _tar_dist_max = self.state_machine_conditions[task]['tar_dist_range']
        rand_dist = (_tar_dist_max - _tar_dist_min) * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device) + _tar_dist_min
        
        rand_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        if task == STATES.SIT:
            rand_theta = -torch.pi*0.5 * torch.ones_like(rand_theta)
        elif task == STATES.STRIKE_PUNCH:
            rand_theta = +torch.pi*0.33 * torch.ones_like(rand_theta)
        self._target_states[env_ids, self.actor_id_offsets.index(task), 0] = rand_dist * torch.cos(rand_theta) + self._humanoid_root_states[env_ids, 0]
        self._target_states[env_ids, self.actor_id_offsets.index(task), 1] = rand_dist * torch.sin(rand_theta) + self._humanoid_root_states[env_ids, 1]
        self._target_states[env_ids, self.actor_id_offsets.index(task), 2] = 0.

        rand_rot_theta = rand_theta - np.pi # [-pi, pi]
        rand_rot_theta += np.pi/2. 
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_states.dtype, device=self._target_states.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis)

        self._target_rot_theta[env_ids,  self.actor_id_offsets.index(task)] = rand_rot_theta
        # self._target_rotmat[env_ids] = angle_to_2d_rotmat(self._target_rot_theta[env_ids])
        self._target_states[env_ids, self.actor_id_offsets.index(task), 3:7] = rand_rot
        self._target_states[env_ids, self.actor_id_offsets.index(task), 7:10] = 0.0
        self._target_states[env_ids, self.actor_id_offsets.index(task), 10:13] = 0.0
        return
    
    def _reset_reach_target(self, env_ids):
        # self.set_random_target_loc(env_ids, STATES.REACH)
        # FIXME - this code is DEMO SPECIFIC - reach near the strike target:
        strike_states = self._target_states[env_ids, self.actor_id_offsets.index(STATES.STRIKE_PUNCH)].clone()
        strike_states[:, :2] -= 0.8
        self._target_states[env_ids, self.actor_id_offsets.index(STATES.REACH)] = strike_states
        self._tar_pos[env_ids, self.actor_id_offsets.index(STATES.REACH), :] =  self._target_states[env_ids, self.actor_id_offsets.index(STATES.REACH), 0:3].clone()
        return
    
    def _reset_strike_target(self, env_ids):
        self.set_random_target_loc(env_ids, STATES.STRIKE_PUNCH)

        self._tar_pos[env_ids, self.actor_id_offsets.index(STATES.STRIKE_PUNCH), :] =  self._target_states[env_ids, self.actor_id_offsets.index(STATES.STRIKE_PUNCH), 0:3].clone()
        self._tar_pos[env_ids, self.actor_id_offsets.index(STATES.STRIKE_PUNCH), -1] += 0.9  # TODO - conditioned on state (kick/punch)
        return
    
    
    def _reset_bench_target(self, env_ids):
        self.set_random_target_loc(env_ids, STATES.SIT)

        self._tar_pos[env_ids, self.actor_id_offsets.index(STATES.SIT), :] =  self._target_states[env_ids, self.actor_id_offsets.index(STATES.SIT), 0:3].clone()  # Guy - for goal reaching
        self._tar_pos[env_ids, self.actor_id_offsets.index(STATES.SIT), -1] += 0.4  # bench height
      
        # self._aux_tar_pos[env_ids] = self._tar_pos[env_ids].clone()  # Marks the place for the head when lying down
        # self._aux_tar_pos[env_ids, 0] += self.orth_arm_delta * torch.sin(self._target_rot_theta[env_ids]) + self.perp_arm_delta * torch.cos(self._target_rot_theta[env_ids])
        # self._aux_tar_pos[env_ids, 1] += -self.orth_arm_delta * torch.cos(self._target_rot_theta[env_ids]) + self.perp_arm_delta * torch.sin(self._target_rot_theta[env_ids])
        # self._aux_tar_pos[env_ids, 2] += self.height_arm_delta

        vec_perp = torch.cat([self.sit_dims[1] / 2. * torch.cos(self._target_rot_theta[env_ids, self.actor_id_offsets.index(STATES.SIT)])[:, None], self.sit_dims[1] / 2. * torch.sin(self._target_rot_theta[env_ids, self.actor_id_offsets.index(STATES.SIT)])[:, None]], dim=-1)
        vec_orth = torch.cat([self.sit_dims[0] / 2. * torch.sin(self._target_rot_theta[env_ids, self.actor_id_offsets.index(STATES.SIT)])[:, None], -self.sit_dims[0] / 2. * torch.cos(self._target_rot_theta[env_ids, self.actor_id_offsets.index(STATES.SIT)])[:, None]], dim=-1)
        self.bench_corners[env_ids, 0, :2] = self._tar_pos[env_ids, self.actor_id_offsets.index(STATES.SIT), :2] + vec_perp + vec_orth
        self.bench_corners[env_ids, 1, :2] = self._tar_pos[env_ids, self.actor_id_offsets.index(STATES.SIT), :2] - vec_perp + vec_orth
        self.bench_corners[env_ids, 2, :2] = self._tar_pos[env_ids, self.actor_id_offsets.index(STATES.SIT), :2] - vec_perp - vec_orth
        self.bench_corners[env_ids, 3, :2] = self._tar_pos[env_ids, self.actor_id_offsets.index(STATES.SIT), :2] + vec_perp - vec_orth

        return

    def get_ids_by_task(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        all_ids = torch.zeros_like(self.is_done)
        all_ids[env_ids] = True
        return {
             _state: (all_ids & (self.cur_state == _state)).nonzero().squeeze(-1) for _state in STATES
        }

    def get_bench_cur_done(self, env_ids):
        task_ids = torch.zeros_like(self.is_done)
        task_ids[env_ids] = True

        in_bench_box = is_point_in_rectangle_vectorized(points=self._rigid_body_pos[:, 0, :2], rect_corners=self.bench_corners[..., :2])
        pelvis_height = self._rigid_body_pos[:, 0, 2]
        in_bench_height = torch.logical_and(self.sit_dims[2] <= pelvis_height, pelvis_height <= self.sit_dims[2] + self.max_gap_from_sit)
        done_sitting = in_bench_box & in_bench_height


        # cur_loc = self._rigid_body_pos[:, self.target_idx[0], self.target_coordinates[0]]
        cur_loc = self._rigid_body_pos[:, mujoco_2_smpl[HML_JOINT_NAMES.index('pelvis')]]
        target_loc = self._tar_pos[:, self.actor_id_offsets.index(STATES.SIT)]
        done_getup = torch.linalg.vector_norm(target_loc-cur_loc, dim=-1) <= self.done_dist

        # # FIXME - Test this (done_sitting might be too strict)
        # head_loc = self._rigid_body_pos[:, mujoco_2_smpl[HML_JOINT_NAMES.index('head')]]
        # arm_loc = self._aux_tar_pos  # [:, self.target_coordinates[0]]
        # done_liedown = torch.linalg.vector_norm(head_loc-arm_loc, dim=-1) <= self.done_dist
        # # done_liedown = done_liedown & done_sitting  # put the head in the right place while the pelvis is on the bench

        done = torch.where(self.cur_state == STATES.SIT, done_sitting, done_getup)
        done = ((self.cur_state == STATES.SIT) & done_sitting) | ((self.cur_state == STATES.GET_UP) & done_getup)
        done = done & task_ids
        return done

    def get_strike_cur_done(self, env_ids):
        task_ids = torch.zeros_like(self.is_done)
        task_ids[env_ids] = True

        # Done if the box is on the ground
        tar_rot = self._target_states[:, self.actor_id_offsets.index(STATES.STRIKE_PUNCH), 3:7]
        up = torch.zeros_like(tar_rot[:, :-1])
        up[..., -1] = 1
        tar_up = quat_rotate(tar_rot, up)
        tar_rot_err = torch.sum(up * tar_up, dim=-1)
        done_strike = tar_rot_err <= self.box_fall_ang

        cur_loc = self._rigid_body_pos[:, mujoco_2_smpl[HML_JOINT_NAMES.index('pelvis')], :2]
        target_loc = self._tar_pos[:, self.actor_id_offsets.index(STATES.STRIKE_PUNCH), :2]  # [:, self.target_coordinates[0]]
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
            target_loc = self._tar_pos[:, self.actor_id_offsets.index(STATES.REACH)].clone()
            target_loc[self.is_2d_target[:, 0].to(bool), -1] = 0
        else:
            cur_loc = self._rigid_body_pos[:, self.target_idx[0], self.target_coordinates[0]]
            target_loc = self._tar_pos[:, self.actor_id_offsets.index(STATES.REACH), self.target_coordinates[0]]
        done = torch.linalg.vector_norm(target_loc-cur_loc, dim=-1) <= self.done_dist
        return done.clone() & task_ids
    
    def update_bench_state_machine(self, env_ids):
        # If done sitting -> get up
        task_ids = torch.zeros_like(self.is_done)
        task_ids[env_ids] = True

        done_was_n_frames_ago = self.progress_buf - self.last_done
        done_sit = torch.logical_and(done_was_n_frames_ago == 50, self.cur_state == STATES.SIT)  
        done_getup = torch.logical_and(done_was_n_frames_ago == 30, self.cur_state == STATES.GET_UP)  
        _rand = torch.bernoulli(self.getup_prob * torch.ones([self.num_envs], device=done_sit.device, dtype=done_sit.dtype)).to(bool)
        switch_to_getup = done_sit & _rand & task_ids
        switch_to_reach = done_getup & task_ids
        # switch_to_sit = (done_was_n_frames_ago == self.state_transition_time) & self.cur_state == STATES.LIE_DOWN   # LIE_DOWN -> SIT
        all_switch = switch_to_getup | switch_to_reach # | switch_to_sit
        self.cur_state[switch_to_getup] = self.next_state[STATES.SIT]
        if self.next_state[STATES.SIT] != STATES.SIT:
            self.is_done[switch_to_getup] = False  # starting a new task
        self.cur_state[switch_to_reach] = self.next_state[STATES.GET_UP]
        if self.next_state[STATES.GET_UP] != STATES.GET_UP:
            self.is_done[switch_to_reach] = False  # starting a new task        
        
        if torch.any(all_switch):
            self.update_mdm_conditions(all_switch.nonzero())

        if torch.any(switch_to_getup):
            getup_deltas = torch.zeros_like(self._tar_pos[switch_to_getup, self.actor_id_offsets.index(STATES.SIT)])
            getup_deltas[:, 0] = self.getup_distance * torch.sin(self._target_rot_theta[switch_to_getup, self.actor_id_offsets.index(STATES.SIT)])
            getup_deltas[:, 1] = -self.getup_distance * torch.cos(self._target_rot_theta[switch_to_getup, self.actor_id_offsets.index(STATES.SIT)])
            getup_deltas[:, 2] = 0.53  # standing height 0.93
            self._tar_pos[switch_to_getup, self.actor_id_offsets.index(STATES.SIT)] += getup_deltas
        
        return
    
    def update_strike_state_machine(self, env_ids):
        # If done sitting -> get up
        task_ids = torch.zeros_like(self.is_done)
        task_ids[env_ids] = True

        # done_was_n_frames_ago = self.progress_buf - self.last_done
        is_strike = torch.logical_or(self.cur_state == STATES.STRIKE_KICK, self.cur_state == STATES.STRIKE_PUNCH)
        done_strike = self.is_done & is_strike &  task_ids
        self.cur_state[done_strike] = STATES.HALT
        self.is_done[done_strike] = False  # starting a new task
        # print(done_was_n_frames_ago, self.cur_state)
        
        if torch.any(done_strike):
            self.update_mdm_conditions(done_strike.nonzero())

        if torch.any(done_strike):
            halt_pose = self._tar_pos[done_strike, self.actor_id_offsets.index(STATES.REACH)].clone()
            self._tar_pos[done_strike, self.actor_id_offsets.index(STATES.STRIKE_PUNCH)] = halt_pose
        return
    
    def update_reach_state_machine(self, env_ids):
        task_ids = torch.zeros_like(self.is_done)
        task_ids[env_ids] = True

        # done_was_n_frames_ago = self.progress_buf - self.last_done
        is_reach = self.cur_state == STATES.REACH
        done_reach = self.is_done & is_reach &  task_ids
        self.cur_state[done_reach] = self.next_state[STATES.REACH]
        if self.next_state[STATES.REACH] != STATES.REACH:
            self.is_done[done_reach] = False  # starting a new task
        
        if torch.any(done_reach):
            self.update_mdm_conditions(done_reach.nonzero())

        return
    
    
    def _draw_task(self):
        if self.support_phc_markers:
            self._update_marker()
        
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        # starts = self._rigid_body_pos[:, self._reach_body_id, :]
        
        
        
        starts = self._rigid_body_pos[:, mujoco_2_smpl[HML_JOINT_NAMES.index('pelvis')], :]
        tar_idx = torch.tensor([self.state_to_tar_idx[int(s)] for s in self.cur_state], device=self.device, dtype=torch.int64)
        ends = self._tar_pos[torch.arange(self.num_envs), tar_idx]

        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        
        # Debug goal reaching MDM part
        if hasattr(self, 'debug_starts'):
            _max_n_joints = 2
            _color = np.array([[0., 0., 1.], [1., 0., 0.]], dtype=np.float32)  # blue, red
            _verts = [torch.cat([self.debug_starts[:, i], self.debug_ends[:, i]], dim=-1).cpu().numpy() for i in range(_max_n_joints)]
            # print(self.debug_starts.shape, self.debug_ends.shape, _verts.shape, verts.shape)
            for env_i, env_ptr in enumerate(self.envs):
                for joint_i in range(_max_n_joints):
                    if joint_i < self.num_target_joints[env_i]:
                        curr_verts = _verts[joint_i][env_i]
                        curr_verts = curr_verts.reshape([1, 6])
                        self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, _color[joint_i])
        

        
        if hasattr(self, 'bench_corners'):
            _color = np.array([1., 0., 0.], dtype=np.float32)  # red
            for _rec_i in range(2):
                _bench_corners = self.bench_corners.clone()
                _bench_corners[..., -1] += (_rec_i * self.max_gap_from_sit)
                for env_i, env_ptr in enumerate(self.envs):
                    if self.init_state_per_env[env_i] == STATES.SIT:
                        for i in range(4):
                            _verts = torch.cat([_bench_corners[env_i, i], _bench_corners[env_i, (i+1)%4]], dim=-1).cpu().numpy()[None]
                            self.gym.add_lines(self.viewer, env_ptr, _verts.shape[0], _verts, _color)
        
        
        # color characters according to done signal
        done_color = np.array([0.8, 0.8, 0.8])  # grey
        not_done_color = np.array([0.2, 0.2, 0.8])  # blue
        done_ids = self.is_done.nonzero().to(self.device)
        not_done_ids = (~self.is_done).nonzero().to(self.device)
        self.set_char_color(done_color, done_ids)
        self.set_char_color(not_done_color, not_done_ids)
        
        return
    