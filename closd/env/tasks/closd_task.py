from closd.env.tasks import closd

from isaacgym.torch_utils import *
from isaacgym import gymapi
from closd.utils.closd_util import STATES
from closd.diffusion_planner.data_loaders.humanml_utils import HML_JOINT_NAMES, HML_EE_JOINT_NAMES
from closd.utils.rep_util import mujoco_2_smpl

import random

# base class for AdapNet fine-tuning with aux task (additional to imitation)
class CLoSDTask(closd.CLoSD):


    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self.init_state = STATES.NO_STATE
        self.has_mdm_task = True
        self.has_aux_task = False
        self.is_done = torch.zeros([self.num_envs], dtype=torch.bool).cuda()
        self.last_done = torch.ones([self.num_envs], dtype=torch.int64).cuda() * self.max_episode_length
        self.done_dist = cfg["env"]["done_dist"]
        self.lr_strings = ['left', 'right']
        self.state_transition_time = 0 # 0 [sec] # number of frames between done was triggered to the state transition  # default config

        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._aux_tar_pos =  torch.zeros_like(self._tar_pos)  # A place holder for bench, not in use here

        # FIXME - instead of fixed text, use a dataset per state
        self.state_machine_conditions = {
            STATES.NO_STATE: {'heading': True, 
                              'joint_names': ['traj'],
                              'text': 'A person is walking.'},
            STATES.REACH: {'heading': True,  # True, 
                           'joint_names': ['traj'],
                           'tar_dist_range': [1., 3.],
                           'text': 'A person is walking.'},
            STATES.STRIKE_KICK: {'heading': True, 
                            'joint_names': ['*_foot'],
                            'tar_dist_range': [1., 1.5],
                            'text': 'A person is performing a high kick with the *.',},  # * is a placeholder for left/right
            STATES.STRIKE_PUNCH: {'heading': True, 
                'joint_names': ['*_wrist'],
                'tar_dist_range': [1., 1.5],
                'text': 'A person is punching with the * hand.',},  # * is a placeholder for left/right
            STATES.HALT: {'heading': True,
                'joint_names': ['traj'],
                'text': 'A person is standing still.',},
            STATES.SIT: {'heading': True, 
                         'joint_names': ['pelvis'],
                         'tar_dist_range': [1., 1.2],
                         'text': 'A person is sitting down on a bench.',},
            STATES.GET_UP: {'heading': True, 
                            'joint_names': ['pelvis'],
                            'text': 'A person is getting up from a bench and standing.',},
            STATES.TEXT2MOTION: {'heading': False, 
                            'joint_names': [],
                            'text': '',},
        }

        # MDM conditions - init with placeholders
        self.all_goal_joint_names = ['pelvis'] + HML_EE_JOINT_NAMES
        self.extended_goal_joint_names = self.all_goal_joint_names + ['traj', 'heading']
        self.num_joint_conditions = len(self.extended_goal_joint_names)
        self.hml_prompts = [''] * self.num_envs  # would like to use an np.array but then the text lentgh has to be set in advance
        self.hml_tokens = [''] * self.num_envs  # would like to use an np.array but then the text lentgh has to be set in advance
        self.db_keys = [''] * self.num_envs  # would like to use an np.array but then the text lentgh has to be set in advance
        self.hml_lengths = np.empty(self.num_envs, dtype=int)  # lengths is not used by DiMP, it is just saved as an additional info when saving the episode
        self.is_heading = torch.zeros([self.num_envs], dtype=torch.bool).cuda()
        self.num_target_joints = torch.zeros([self.num_envs], dtype=torch.int64).cuda()
        self.is_2d_target = torch.zeros([self.num_envs, 2], dtype=torch.int64).cuda()  # used as an index
        self.mujoco_joint_idx = torch.zeros([self.num_envs, 2], dtype=torch.int64).cuda()  # 2 is the maximal supprted joints
        self.cur_joint_condition = [[]] * self.num_envs
        # self.multi_target_data = torch.zeros([self.num_envs, self.num_joint_conditions, 3], dtype=torch.float32).cuda()

        self.support_phc_markers = self.cfg['env']['support_phc_markers']
        if not self.headless and self.support_phc_markers:
            self._build_marker_state_tensors()

        return

    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        if not self.headless and self.support_phc_markers:
            self._build_marker(env_id, env_ptr)

        return

    def _build_marker(self, env_id, env_ptr):
        super()._build_marker(env_id, env_ptr)
        col_group = env_id
        col_filter = 2
        segmentation_id = 0

        default_pose = gymapi.Transform()
        
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset_aux, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._aux_marker_handles.append(marker_handle)

        return   
    
    def pre_physics_record_states(self):
        self.prev_done = self.is_done.clone()
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        self.support_phc_markers = self.cfg['env']['support_phc_markers']
        if not self.headless and self.support_phc_markers:
            self._aux_marker_handles = []
            self._load_marker_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
    
    def _load_marker_asset(self):
        super()._load_marker_asset()
        asset_root = "closd/data/assets/mjcf"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset_aux = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        assert self.init_state != STATES.NO_STATE or hasattr(self, 'init_state_per_env')  # assert that a valid init state was asigned by the child class
        
        if hasattr(self, 'init_state_per_env'):
            self.cur_state[env_ids] = self.init_state_per_env[env_ids]
        elif type(self.init_state) == list:  # choose randomly from list
            self.cur_state[env_ids] = torch.tensor(random.choices(self.init_state, k=len(env_ids)), 
                                                   dtype=self.cur_state.dtype, device=self.cur_state.device)
        else:
            self.cur_state[env_ids] = self.init_state

        self.update_mdm_conditions(env_ids)

    
    def update_mdm_conditions(self, env_ids):    
        # Must be called after any state update!
        # Creates the static args that will be pluged into MDM's model_kwargs
        # FIXME - Can we avoid the loop?
        for i in env_ids:
            _state = int(self.cur_state[int(i)])
            self.hml_prompts[int(i)] = self.state_machine_conditions[_state]['text']
            _joint_names = self.state_machine_conditions[_state]['joint_names']
            if '*' in self.hml_prompts[int(i)]:
                _direction = random.choice(self.lr_strings)
                self.hml_prompts[int(i)] = self.hml_prompts[int(i)].replace('*', _direction)
                _joint_names = [e.replace('*', _direction) for e in _joint_names]
            self.num_target_joints[int(i)] = len(_joint_names)
            self.is_heading[int(i)] = self.state_machine_conditions[_state]['heading']
            self.cur_joint_condition[int(i)] = _joint_names
            self.is_2d_target[int(i), :self.num_target_joints[int(i)]] = torch.tensor([j == 'traj' for j in _joint_names])
            _joint_names = ['pelvis' if j == 'traj' else j for j in _joint_names]
            self.mujoco_joint_idx[int(i), :self.num_target_joints[int(i)]] = torch.tensor([mujoco_2_smpl[HML_JOINT_NAMES.index(j)] for j in _joint_names])
        return
    
    def post_physics_step(self):
        super().post_physics_step()
        # propagating out for wandb success rate report:
        self.extras['task_done'] = self.is_done.clone()   ###.cpu().numpy()  
        self.extras['task_state'] = self.cur_state.clone()   #.cpu().numpy()
        return
    
    def get_aux_task_obs_size(self):
        return 0
    
    def _update_task(self):
        self.update_done()  # update all envs
        self.update_state_machine()
        # DEBUG
        # print('is_done', self.is_done)
        # # print('progress_buf', self.progress_buf)
        # import pdb
        # if self.is_done[0] and not self.prev_done[0]:
        #     pdb.set_trace()
        # if self.progress_buf[0] == 50:
        #     bbb = gymapi.UsdExportOptions()
        #     bbb.single_file = True
        #     usd_exp = self.gym.create_usd_exporter(bbb)
        #     aaa = self.gym.export_usd_sim(usd_exp, self.sim, 'try.usd')
        #     # exit()
        return
    
    def get_cur_done(self):
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
        return done.clone()
    
    def update_done(self):
        # According to done_dist
        self.is_done = torch.logical_or(self.is_done, self.get_cur_done())
        self.is_done = torch.logical_and(self.is_done, torch.logical_not(self.reset_buf))
        self.is_done = torch.where(self.progress_buf == 0, torch.zeros_like(self.is_done), self.is_done)  # reset at the begining of the episode
        is_new_done = torch.nonzero(torch.logical_and(self.is_done, torch.logical_not(self.prev_done)) == True)
        self.last_done[is_new_done] = self.progress_buf[is_new_done]
        # print(self.last_done, self.is_done)
    
    def update_state_machine(self):
        # Placeholder
        return

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs) and not self.cfg['env']['disable_adaptnet']:
            task_obs_size = self.get_aux_task_obs_size()
            # 2D obs case:
            if type(task_obs_size) == tuple and len(task_obs_size) == 2:
                task_obs_size = task_obs_size[0] * task_obs_size[1]
            obs_size += task_obs_size
        return obs_size
    
    def _compute_aux_reward(self, actions):
        return torch.zeros(self.num_envs, device=self.device, dtype=self._rigid_body_pos.dtype)


    def calc_cur_target_multi_joint(self, max_dist=1.2):
        # FIXME - currently supporting first tartget only
        # returns target [n_envs, 3] cuted for max_dist [m] from current location
        all_start_loc = []
        all_end_loc = []
        all_tars = [self._tar_pos, self._aux_tar_pos]
        for joint_i in range(2):
            start_loc = self._rigid_body_pos[torch.arange(self.num_envs), self.mujoco_joint_idx[:, joint_i]]
            end_loc = all_tars[joint_i].clone()
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

    def calc_cur_target(self, max_dist=1.2):
        # FIXME - currently supporting first tartget only
        # returns target [n_envs, 3] cuted for max_dist [m] from current location
        end_loc = self._tar_pos[:, self.target_coordinates[0]].clone()
        end_loc -= self._rigid_body_pos[:, self.target_idx[0], self.target_coordinates[0]]
        traj_length = torch.linalg.norm(end_loc, dim=-1)[:, None]
        needs_shortening = (traj_length > max_dist).float()
        shortening_factor = (traj_length*needs_shortening/max_dist) + (1.*(1. - needs_shortening))
        end_loc /= shortening_factor
        end_loc += self._rigid_body_pos[:, self.target_idx[0], self.target_coordinates[0]]
        if len(self.target_coordinates[0]) == 2:
            end_loc = torch.cat([end_loc, self._tar_pos[:, [2]]], dim=-1)
        end_loc = end_loc[:, None]  # for maintaining support for multi-targets

        # update marker
        self.debug_starts = self._rigid_body_pos[:, [self.target_idx[0]]].clone()
        self.debug_ends = end_loc.clone()

        # print('debug_ends', self.debug_ends)

        return end_loc
    
    # def get_line_condition(self, translated_end_loc):

    #     start_loc = torch.zeros_like(translated_end_loc)
    #     end_heading = (torch.atan2(translated_end_loc[:, 0], translated_end_loc[:, 1])[:, None, None] - torch.pi)  % (2*torch.pi) - torch.pi  # [-pi, pi]            
    #     start_heading = torch.zeros_like(end_heading)

    #     # Calc traj
    #     _t = torch.linspace(0., 1., self.mdm.model.pred_len+1, device=self.device)  # assuming fixed pred length
    #     traj = start_loc[:, :, None] + _t[None, None, :] * (translated_end_loc[:, :, None] - start_loc[:, :, None])
    #     heading_traj = start_heading + _t[None, None, :] * (end_heading - start_heading)  # FIXME - make it non-linear
    #     motion_shape = (self.num_envs, self.mdm.njoints, self.mdm.nfeats, self.mdm.model.pred_len)
    #     inpainted_motion = torch.zeros(motion_shape, dtype=torch.float32,
    #                                             device=self.device)  # True means use gt motion
    #     traj_data_unnorm = traj_global2vel(traj, heading_traj).to(self.device)
    #     inpainted_motion[:, :3] = (traj_data_unnorm - self.mean[:3][None, :, None, None]) / self.std[:3][None, :, None, None]
        
    #     inpainting_mask = torch.zeros(motion_shape, dtype=torch.bool,
    #                                             device=self.device)  # True means use gt motion
    #     inpainting_mask[:, :3] = True

    #     return {'condition_mask': inpainting_mask,
    #             'condition_input': inpainted_motion,
    #             }


    def _draw_task(self):
        if self.support_phc_markers:
            self._update_marker()
        
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        # starts = self._rigid_body_pos[:, self._reach_body_id, :]
        starts = self._rigid_body_pos[:, mujoco_2_smpl[HML_JOINT_NAMES.index('pelvis')], :]
        ends = self._tar_pos

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
