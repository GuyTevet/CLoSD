import os
import torch
import numpy as np
import json
import time
import pickle
import yaml
from closd.utils.flags import flags
from datetime import datetime
from isaacgym import gymtorch
from scipy.spatial.transform import Rotation as sRot
from isaacgym.torch_utils import *

from closd.env.tasks.humanoid import Humanoid
import closd.env.tasks.humanoid_im as humanoid_im

from closd.diffusion_planner.utils.fixseed import fixseed
from closd.diffusion_planner.utils.model_util import create_model_and_diffusion, load_saved_model
from closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel
from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import recover_from_ric
from closd.diffusion_planner.data_loaders.humanml.utils.plot_script import plot_3d_motion
from closd.diffusion_planner.sample.generate import save_multiple_samples
from closd.diffusion_planner.utils import cond_util
from closd.diffusion_planner.data_loaders.humanml_utils import HML_JOINT_NAMES
from closd.diffusion_planner.data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain

from closd.utils.rep_util import mujoco_2_smpl
from closd.utils.rep_util import RepresentationHandler
from closd.utils.closd_util import STATES
from closd.utils.rep_util import smpl_2_mujoco
from closd.utils.poselib.poselib.skeleton.skeleton3d import SkeletonMotion
# from closd.phc_utils import torch_utils
# from closd.phc_utils.filter_utils import gaussian_filter1d_torch, apply_one_side_filter1d_torch


class FakeMDMArgs:
    def __init__(self, input_path):
        self.predict_eps = False # default
        with open(input_path) as fr:
            json_dict = json.load(fr)
        for k, v in json_dict.items():
            setattr(self, k, v)


class CLoSD(humanoid_im.HumanoidIm):


    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        # DEMO - INIT START        
        self.local_translation_batch = self.skeleton_trees[0].local_translation[None,]
        self.parent_indices = self.skeleton_trees[0].parent_indices
        self.limb_pairs = torch.cat([self.parent_indices[1:6][None], torch.arange(1,6)[None]], dim=0) # 5 limbs: [parent, child]
        self.pose_mat = torch.eye(3).repeat(self.num_envs, 24, 1, 1).to(self.device)
        self.trans = torch.zeros(self.num_envs, 3).to(self.device)

        self.prev_ref_body_pos = torch.zeros(self.num_envs, 24, 3).to(self.device)
        self.prev_ref_body_rot = torch.zeros(self.num_envs, 24, 4).to(self.device)

        self.zero_trans = torch.zeros([self.num_envs, 3])
        self.s_dt = 1 / 30

        self.to_isaac_mat = torch.from_numpy(sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()).float().cuda()
        self.to_global = torch.from_numpy(sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv().as_matrix()).float().cuda()

        # self.root_pos_acc = TorchBuffer(n_envs=self.num_envs, buff_size=3, device=self.device) # deque(maxlen=30)
        # self.body_pos_acc = TorchBuffer(n_envs=self.num_envs, buff_size=72, device=self.device) # deque(maxlen=30)
        # self.body_rot_acc = deque(maxlen=30)

        # flags.no_collision_check = True  # practicaly disable when humanoid falls - not good for training
        flags.show_traj = True
        self.close_distance = 0.5
        # self.mean_limb_lengths = np.array([0.1061, 0.3624, 0.4015, 0.1384, 0.1132], dtype=np.float32)[None, :]
        self.mean_limb_lengths = torch.tensor([0.1061, 0.3624, 0.4015, 0.1384, 0.1132], dtype=torch.float32)[None, :].cuda()

        # self.filter_sigmas = [2, 5, 10]
        # self.filter_kernels = {sig: gaussian_filter1d_torch(sigma=sig).cuda()for sig in self.filter_sigmas}

        assert self.obs_v == 7
        # DEMO - INIT END    

        self.frame_idx = 0
        self.last_gen_idx = -1

        self.time_prints = False  # Hardcoded
        self.mdm_path = self.cfg['env']['dip']['model_path']
        self.mdm_args_path = os.path.join(os.path.dirname(self.mdm_path), 'args.json')
        self.fake_mdm_args = FakeMDMArgs(self.mdm_args_path)
        hml_mode = 'train'  # we would like to get the full lengths so cannot use 'text_only'
        split = 'test' if self.cfg.test else 'train'
        self.mdm_data = get_dataset_loader(name=self.fake_mdm_args.dataset,
                              batch_size=self.num_envs,
                              num_frames=196,  # this argument is not used by the hml dataset. but must send a value so sending max hml motion length.
                              split=split,
                              hml_mode=hml_mode,
                              drop_last=False,
                              return_keys=True)
        self.full_mdm_dataset_size = len(self.mdm_data.dataset)
        print(f'MDM dataset size: {self.full_mdm_dataset_size}')
        self.mdm_data_iter = iter(self.mdm_data)
        self.mdm, self.diffusion = create_model_and_diffusion(self.fake_mdm_args, self.mdm_data)
        self.sample_fn = self.diffusion.p_sample_loop
        fixseed(self.fake_mdm_args.seed)
        print(f"Loading checkpoints from [{self.mdm_path}]...")
        load_saved_model(self.mdm, self.mdm_path, use_avg=self.fake_mdm_args.use_ema)
        self.spatial_condition = self.fake_mdm_args.__dict__.get('spatial_condition', None)
        self.multi_target_cond = self.fake_mdm_args.__dict__.get('multi_target_cond', False)

        self.context_switch_prob = self.cfg['env']['dip']['context_switch_prob']
        self.planning_horizon_multiplyer = self.cfg['env']['dip']['planning_horizon_multiplyer']
        self.mdm_cfg_param = self.cfg['env']['dip']['cfg_param']
        if self.mdm_cfg_param != 1.:
            self.mdm = ClassifierFreeSampleModel(self.mdm)   # wrapping model with the classifier-free sampler

        self.mdm_device = 'cuda:0'
        self.mdm.to(self.mdm_device)
        self.mdm.eval()  # disable random masking
        
        self.mean = torch.from_numpy(self.mdm_data.dataset.t2m_dataset.mean).cuda()
        self.std = torch.from_numpy(self.mdm_data.dataset.t2m_dataset.std).cuda()

        self.real_mesh = False  # add trrain if false

        self.cur_state = torch.ones([self.num_envs], dtype=torch.int64).cuda() * STATES.NO_STATE  # Placeholder for state machine

        # TODO - GUY - CONSTS MOVE TO CONFIG
        # TODO - GUY - FIGURE OUT ROUNDING VS //
        assert self.mdm.is_prefix_comp
        print(f'Overwriting [context_len, pred_len] to [{self.fake_mdm_args.context_len}, {self.fake_mdm_args.pred_len}] according to the prefix completion MDM parameters.')
        self.max_frame_20fps = self.fake_mdm_args.pred_len
        self.max_frame_30fps = self.max_frame_20fps * 30 // 20

        self.context_len_20fps = self.fake_mdm_args.context_len
        self.context_len_30fps = int(round(self.context_len_20fps * 30 / 20))  # we use int(round()) so fps conversion forth and back will be consistent

        self.planning_horizon_20fps = min(self.cfg['env']['dip']['planning_horizon'], self.fake_mdm_args.pred_len)
        self.planning_horizon_30fps = self.planning_horizon_20fps * 30 // 20

        self.pose_buffer = torch.empty([self.num_envs, self.context_len_30fps, 24, 3], dtype=torch.float32, device=self.device)
        self.planning_horizon = torch.empty([self.num_envs, self.planning_horizon_30fps, 24, 3], dtype=torch.float32, device=self.device)
        self.used_db_keys = []  # book which motion keys have been already used, i.e., their text used for generation

        # save episodes in hml format, to be used for evaluation
        self.init_save_hml_episodes()

        self.mdm_tensor_shape = (self.cfg['env']['num_envs'], self.mdm.njoints, self.mdm.nfeats, self.max_frame_20fps)

        # init rep handler
        self.rep = RepresentationHandler(mean=self.mean,
                                         std=self.std,
                                         init_humanoid_root_pose=self._rigid_body_pos[:, 0],
                                         time_prints=self.time_prints,
                                         )
        
    def init_save_hml_episodes(self):
        save_motions = self.cfg['env'].get('save_motion', {})
        self.save_hml_episodes = save_motions.get('save_hml_episodes', False)
        # self.save_hml_episodes = self.cfg['env']['save_motion']['save_hml_episodes']  # this line crashes when "env" is not "closd_t2m". fixed above.
        if self.save_hml_episodes:
            
            self.max_saved_hml_episodes = self.cfg['env']['save_motion']['max_saved_hml_episodes']
            self.save_episodes_file_name = 'CLoSD'
            self.n_all_episodes = 0
            # the amount of frames to crop from the beginningof the episode
            self.prefix_crop_size = self.cfg['env']['save_motion']['prefix_crop_size']
            # assert self.prefix_crop_size >= self.context_len_30fps
            self.max_episode_length = self.max_episode_length + self.prefix_crop_size  # FIXME: it is bad practice to overide a global variable. better use a different name
            self.save_name_suffix = self.cfg['env']['save_motion']['save_name_suffix']
            self.done_saving_episodes = False  # FIXME: convert to a boolean in the base class, saying "task_done"

            # prepare buffers where episode data will be stored. Use PHC format to make all conversions together at the end only
            self.episode_buf = torch.empty([self.num_envs,                self.max_episode_length, 24, 3], dtype=torch.float32, device=self.device)
            self.all_episodes = torch.empty([self.max_saved_hml_episodes, self.max_episode_length-self.prefix_crop_size, 24, 3], dtype=torch.float32, device='cpu')  # use 'cpu' as gpu has smaller space
            self.all_episodes_prompts = [None] * self.max_saved_hml_episodes
            self.all_episodes_tokens = [None] * self.max_saved_hml_episodes
            self.all_episodes_db_keys = [None] * self.max_saved_hml_episodes
            self.all_episodes_lengths = np.empty(self.max_saved_hml_episodes)           
            
            # create output folder
            self.save_hml_episodes_dir = self.cfg['env']['save_motion']['save_hml_episodes_dir']
            if self.save_hml_episodes_dir in ['', None]:
                model_path_no_ext = os.path.splitext(self.cfg['env']['dip']['model_path'])[0]
                now = datetime.now()
                timestamp = now.strftime("%y_%m_%d_%H_%M")
                save_name_suffix = self.save_name_suffix if not self.save_name_suffix in ['',None] else timestamp
                self.save_hml_episodes_dir = f'{model_path_no_ext}.CLoSD_cfg{self.mdm_cfg_param}_planh{self.planning_horizon_20fps}_crop{self.prefix_crop_size}_exp{self.cfg["exp_name"]}_epoc{self.cfg["epoch"]}_max{self.max_saved_hml_episodes}_{save_name_suffix}'
            os.makedirs(self.save_hml_episodes_dir)
            print(f'Saving hml episodes to [{self.save_hml_episodes_dir}]')
            
            # save config data
            cfg_file_name = os.path.join(self.save_hml_episodes_dir, 'config.yaml')
            with open(cfg_file_name, 'w') as cfg_file:
                yaml.dump(self.cfg, cfg_file, default_flow_style=False)
    

    def build_completion_input(self, context_switch_vec=None):
        # input: hml_poses [n_envs, n_frames@20fps, 263]
        # context_switch_vec [n_envs] if not None - indicates which env will use prediction context insted of sim contest
        # output: 
        #   inpainted_motion [bs, 263, 1, max_frames] where hml_poses is the prefix and the rest is zeros
        #   inpainting_mask [bs, 263, 1, max_frames] - true only for the prefix frames

        pose_context = self.pose_buffer  # self.pose_buffer contains last 20 sim poses (for prefix_len=20)
        if self.cfg['env']['dip']['limit_context'] is not None:
            pose_context = pose_context[:, -self.cfg['env']['dip']['limit_context']:]  

        aux_points = None  
        if self.multi_target_cond:
            aux_points = self.calc_cur_target_multi_joint()  # [bs, n_points, 3]     
        
        # Real performed motions from the simulator, translated to HML format
        sim_context, translated_aux_points, recon_data = self.rep.pose_to_hml(pose_context, aux_points, fix_ik_bug=True)  # [bs, n_frames@20fps, 263], [bs, n_points, 3]
        

        sim_context = sim_context.unsqueeze(2).permute(0, 3, 2, 1)  # [bs, 263, 1, n_frames@20fps]

        if context_switch_vec is not None:
            pred_context = self.cur_mdm_pred[..., -sim_context.shape[-1]:]
            is_pred = context_switch_vec.view(-1, 1, 1, 1)
            hml_context = (is_pred * pred_context) + ((1. - is_pred) * sim_context)
        else:
            hml_context = sim_context

        
        if self.planning_horizon_multiplyer > 1 and self.frame_idx > 0 and self.frame_idx % (self.planning_horizon_30fps * self.planning_horizon_multiplyer) != 0:
            # extand the planning horizon artiffitially. e.g - planning_horizon=40, planning_horizon_multiplyer=2 -> planning horizom will be 80 in practice.
            pred_context = self.cur_mdm_pred[..., -sim_context.shape[-1]:]
            hml_context = pred_context

        context_len = hml_context.shape[-1]
        assert context_len == self.context_len_20fps
        inpainted_motion = torch.zeros(self.mdm_tensor_shape, dtype=torch.float32, device=hml_context.device)
        inpainted_motion[:, :, :, :context_len] = hml_context
        inpainted_motion = inpainted_motion.to(self.mdm_device)
        
        inpainting_mask = torch.zeros(self.mdm_tensor_shape, dtype=torch.bool, device=self.mdm_device)
        inpainting_mask[:, :, :, :context_len] = True  # True means use gt motion

        mask = torch.ones_like(inpainting_mask)[:, [0]]

        aux_entries = {'mask': mask, 'prefix_len': context_len, }

        if self.mdm_cfg_param != 1.:
            aux_entries['scale'] = torch.ones(self.num_envs, device=self.mdm_device) * self.mdm_cfg_param

        # init prefix from humanml real data if exists at the first MDM call
        # used for the text-to-motion task only!
        if hasattr(self, 'hml_prefix_from_data'):
            is_first_iter = self.progress_buf < self.planning_horizon_30fps
            hml_context[is_first_iter] = self.hml_prefix_from_data[is_first_iter, :, :, -self.context_len_20fps:]

        if self.mdm.is_prefix_comp:
            aux_entries.update({'prefix': hml_context})
        else:
            aux_entries.update({'inpainted_motion': inpainted_motion, 'inpainting_mask': inpainting_mask})
        
        if self.multi_target_cond:
            _target_cond = torch.zeros((self.num_envs, self.num_joint_conditions, 3), dtype=translated_aux_points.dtype, device=translated_aux_points.device)
            
            for joint_i in range(2):
                joints_in_use = torch.nonzero(joint_i < self.num_target_joints).squeeze(-1)
                if joints_in_use.shape[0] > 0:
                    _target = translated_aux_points[joints_in_use, joint_i]
                    joint_names_in_use = [self.cur_joint_condition[j][joint_i] for j in joints_in_use]
                    _joint_entry = [self.extended_goal_joint_names.index(j) for j in joint_names_in_use]
                    _target_cond[joints_in_use, _joint_entry] = _target
            
            _target_cond[:, self.extended_goal_joint_names.index('traj'), 1] = 0.   # zero the y axis for the trajectory

            # asign heading
            _target_heading = torch.atan2(translated_aux_points[:, 0, 0], translated_aux_points[:, 0, 2])[:, None]  # heading is according to the first joint
            backward_heading = (self.cur_state == STATES.SIT)
            _target_heading[backward_heading] = _target_heading[backward_heading] % (2*torch.pi) - torch.pi
            _target_cond[self.is_heading, self.extended_goal_joint_names.index('heading'), 0] = _target_heading[self.is_heading][:, 0]

            # update nodel_kwargs
            aux_entries.update({'goal_cond': _target, 'heading_cond': _target_heading})  # for vis
            aux_entries.update({'pred_target_cond': _target_cond, 'target_cond': _target_cond, 'target_joint_names': self.cur_joint_condition, 'is_heading': self.is_heading})
        return aux_entries, recon_data

    
    def get_text_prompts(self):
        if self.hml_prompts[0] != '':  # the single user-defined text prompt case
            return self.hml_prompts
        else:  # get prompts from dataset
            raise ValueError('prompts must be specified, this mode is no longer supported.')
    
    def get_mdm_next_planning_horizon(self):

        context_switch_vec = None
        if self.context_switch_prob > 0. and self.frame_idx > 0:
            context_switch_vec = torch.bernoulli(self.context_switch_prob * torch.ones([self.num_envs], device=self.device, dtype=self._rigid_body_pos.dtype))
        
        cond_fn = None
        aux_entries, recon_data = self.build_completion_input(context_switch_vec)
        
        # Build MDM inputs
        model_kwargs = {'y': {}}
        model_kwargs['y']['text'] = self.get_text_prompts()
        model_kwargs['y']['text_embed'] = self.mdm.encode_text(model_kwargs['y']['text'])  # encoding once instead of each iteration saves lots of time!
        model_kwargs['y'].update(aux_entries)
        init_image = None
        
        # Run MDM with prefix outpainting
        start_time = time.time()

        sample = self.sample_fn(
            self.mdm,
            self.mdm_tensor_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=init_image,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
            cond_fn=cond_fn,
        )  # [bs, 263, 1, 60]
        self.cur_mdm_pred = sample  # used in the context_switch feature
        if self.time_prints:
            print('=== sample mdm for [{}] envs took [{:.2f}] sec'.format(sample.shape[0], time.time() - start_time))

        sample_reshaped = sample.squeeze(2).permute(0, 2, 1)
        sample_xyz = self.rep.hml_to_pose(sample_reshaped, recon_data, sim_at_hml_idx=model_kwargs['y']['prefix_len']-1)  # hml rep [bs, n_frames_20fps, 263] -> smpl xyz [bs, n_frames_30fps, 24, 3]

        if self.cfg['env']['dip']['debug_hml']:
            print(f'in get_mdm_next_planning_horizon: prompts={model_kwargs["y"]["text"][:2]}')
        if self.cfg['env']['dip']['debug_hml']:
            print(f'in get_mdm_next_planning_horizon: prompts={self.hml_prompts[:2]}')
            self.visualize(sample[:1],
                               'mdm_debug/prefixComp_{}_{}.mp4'.format(self.frame_idx, self.hml_prompts[0].replace('.', '').replace(' ', '_')),
                               is_prefix_comp=True, model_kwargs=model_kwargs,)

        # Extract the planning horizon
        context_len_30fps = model_kwargs['y']['prefix_len'] * 30 // 20
        planning_horizon = sample_xyz[:, context_len_30fps-1:context_len_30fps+self.planning_horizon_30fps]  # [x, -z, y]

        return planning_horizon[:, 0], planning_horizon[:, 1:]
    
    
    def get_pred_pose(self):
        # returns next pose to imitate in xyz [bs, 24, 3]
        # if needed, will predict the next 6 poses with the model
        # self.frame_idx = int(self.progress_buf[0])  # assuming all the buffer contains the same value

        # get next motion token if needed
        # the check for safe_reset is required when _reset_envs is called twice from Humanoid.reset() and DB reading is done only in the 1st call
        # FIXME: instead of calling get_mdm_next_planning_horizon with no reason, avoid calling update_mdm_conditions() up a safe_reset flag
        if self.frame_idx % self.planning_horizon_30fps == 0 and (self.last_gen_idx != self.frame_idx or self.safe_reset):  
            # self.clear_buffers()
            self.last_gen_idx = self.frame_idx  # to ensure not generating twice
            start_time = time.time()
            self.prev_real_frame, self.planning_horizon = self.get_mdm_next_planning_horizon()  # [bs, horizon_len, 24, 3]
            if self.time_prints:
                print('=== get_mdm_next_planning_horizon for [{}] envs took [{:.2f}] sec'.format(self.planning_horizon.shape[0], time.time() - start_time))
        

        self.next_pred_pose = self.planning_horizon[:, self.frame_idx % self.planning_horizon_30fps]  # xyz pred [bs, 24, 3]

        json_resp = {
            "j3d": self.next_pred_pose,  # .tolist(),
            "dt": self.dt,
        }

        if self.frame_idx % self.planning_horizon_30fps == 0:
            json_resp.update({"j3d_prev": self.prev_real_frame})

        return json_resp
    
    def _compute_observations(self, env_ids=None):
        # env_ids is used for resetting
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)

        self_obs = self._compute_humanoid_obs(env_ids)
        self.self_obs_buf[env_ids] = self_obs

        if (self._enable_task_obs):
            task_obs, _ = self._compute_task_obs_demo(env_ids)
            obs = torch.cat([self_obs, task_obs], dim=-1)
            if self.has_aux_task:
                if self.get_aux_task_obs_size() != 0:
                    aux_task_obs = self._compute_aux_task_obs(env_ids)
                    obs = torch.cat([obs, aux_task_obs], dim=-1)
        else:
            obs = self_obs

        if self.obs_v == 4:
            # Double sub will return a copy.
            B, N = obs.shape
            sums = self.obs_buf[env_ids, 0:10].abs().sum(dim=1)
            zeros = sums == 0
            nonzero = ~zeros
            obs_slice = self.obs_buf[env_ids]
            obs_slice[zeros] = torch.tile(obs[zeros], (1, 5))
            obs_slice[nonzero] = torch.cat([obs_slice[nonzero, N:], obs[nonzero]], dim=-1)
            self.obs_buf[env_ids] = obs_slice
        else:
            self.obs_buf[env_ids] = obs
        return obs
    
    
    def _reset_pose_buffer(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        elif len(env_ids) == 0:
            return

        # init pose buffer from current performed pose
        body_pos = self._rigid_body_pos[env_ids]  # what's used by isaac
        if self.save_hml_episodes:
            self.episode_buf[env_ids] = body_pos[:, None].repeat(1, self.max_episode_length, 1, 1)
        ref_pos = self.rep.sim_pose_to_ref_pose(body_pos)[:, None]  # convert to what's used by phc
        self.pose_buffer[env_ids] = body_pos[:, None].repeat(1, self.context_len_30fps, 1, 1)  # [n_envs, len(env_ids), n_joints, 3]
        self.planning_horizon[env_ids] = ref_pos.repeat(1, self.planning_horizon_30fps, 1, 1)  # [n_envs, horizon_len, 24, 3]
        # assert (self.rep.sim_pose_to_ref_pose(body_pos) - self.rep.sim_pose_to_ref_pose_old(body_pos)).abs().max() < 1e-6

        # init planning_horizon from humanml real data if exists
        # In used for the text-to-motion task only!
        if hasattr(self, 'hml_prefix_from_data'):
            # convert prefix to isaac format
            # cur_frame = self.rep.sim_pose_to_ref_pose(self._rigid_body_pos)
            sim_at_hml_idx = (self.frame_idx % self.planning_horizon_30fps) * 20 // 30  # FIXME: check if we need a '-1' here
            hml_from_pose, _, recon_data = self.rep.pose_to_hml(self.pose_buffer, fix_ik_bug=True) # compute rot/trans of simulator character

            planning_horizon = self.rep.hml_to_pose(self.hml_prefix_from_data.squeeze(2).permute(0, 2, 1), recon_data, sim_at_hml_idx=sim_at_hml_idx)  # hml rep [bs, n_frames_20fps, 263] -> smpl xyz [bs, n_frames_30fps, 24, 3]

            # place the prefix in the planning_horizon so PHC will start imitating it (until the next call to MDM)
            self.planning_horizon[env_ids] = planning_horizon[env_ids, -self.planning_horizon_30fps:]  # [n_envs, horizon_len, 24, 3]
    
    def _reset_envs(self, env_ids):
        if self.frame_idx == 0:
            self._reset_pose_buffer(env_ids)
        super()._reset_envs(env_ids)
        self._reset_pose_buffer(env_ids)
        return
    
    def post_physics_step(self):
        super().post_physics_step()
        self._update_pose_buffer()
        self.frame_idx += 1
        if self.save_hml_episodes and self.n_all_episodes < self.max_saved_hml_episodes:
            env_ids = self.get_episode_ids()
            self.aggregate_episodes(env_ids)
            if self.n_all_episodes == self.max_saved_hml_episodes:
                self.save_episodes()

    def save_episodes(self):
        self.done_saving_episodes = True
                
        # convert to hml format
        # while fix_ik_bug=False imitates HumanML3D distribution, fix_ik_bug=True attains better quantitative scores, probably because it is closer to the real phisical distribution + it eases the "work" for MDM
        hml_motions, _, _ = self.rep.pose_to_hml(self.all_episodes[:self.n_all_episodes], None, fix_ik_bug=True)  # [bs, max_episode_len@20fps, 263]
        hml_motions = hml_motions.unsqueeze(2).permute(0, 3, 2, 1)  # [bs, 263, 1, max_episode_len@20fps]
            
        data_to_save = {'motion': hml_motions.squeeze().permute(0, 2, 1),  # use the same format used for evaluation
                        'caption': self.all_episodes_prompts[:self.n_all_episodes], 
                        'length': self.all_episodes_lengths[:self.n_all_episodes], 
                        'tokens': self.all_episodes_tokens[:self.n_all_episodes],
                        'db_key': self.all_episodes_db_keys[:self.n_all_episodes]}
                
        file_path = os.path.join(self.save_hml_episodes_dir, self.save_episodes_file_name+'.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=4)
        print(f'Saved episodes to {file_path}')
           
        if self.cfg['env']['dip']['debug_hml'] and hml_motions.shape[0] >= 12:  
            n_vis = 3
            n_files = 4
            for start in range(0,n_files*n_vis, n_vis):
                idx = range(start, start+n_vis)
                vis_path = os.path.join(self.save_hml_episodes_dir, f'{self.save_episodes_file_name}_{start}_{start+n_vis}.mp4')   
                prmopts = np.array(self.all_episodes_prompts)[idx]
                self.visualize(hml_motions[idx].cuda(), vis_path, is_prefix_comp=False, model_kwargs=None, prompts=prmopts)
                print(f'saved visualization to {vis_path}')

    def get_episode_ids(self):
        if_full_episode = self.progress_buf == self.max_episode_length-1
        is_full_valid_episode = torch.logical_and(if_full_episode, torch.logical_not(self._terminate_buf))
        env_ids = torch.nonzero(is_full_valid_episode)  # self._reset_ref_env_ids contains terminations so can't be used
        env_ids = env_ids.squeeze(1).tolist()
        return env_ids

    def aggregate_episodes(self, env_ids):
        if len(env_ids) == 0:
            return
                
        n_episodes_done = len(env_ids)
        idx_from = self.n_all_episodes  
        idx_to = idx_from+n_episodes_done
        if idx_to > self.max_saved_hml_episodes:
            idx_to = self.max_saved_hml_episodes
            env_ids = env_ids[:idx_to-idx_from]
            n_episodes_done = len(env_ids)
        self.all_episodes[idx_from:idx_to] = self.episode_buf[env_ids, self.prefix_crop_size:].cpu()  # use only predicted part w/o the context (actually, w/o what we decided to crop)
        
        self.all_episodes_prompts[idx_from:idx_to] = np.array(self.hml_prompts)[env_ids].tolist()  # must use tolist(), otherwise np dtype comes as a prompt
        self.all_episodes_tokens[idx_from:idx_to] = np.array(self.hml_tokens)[env_ids].tolist()  # must use tolist(), otherwise np dtype comes as a prompt
        self.all_episodes_db_keys[idx_from:idx_to] = np.array(self.db_keys)[env_ids].tolist()  # must use tolist(), otherwise np dtype comes as a prompt
        self.all_episodes_lengths[idx_from:idx_to] = self.hml_lengths[env_ids]
        self.used_db_keys.extend(np.array(self.db_keys)[env_ids])
        self.n_all_episodes += n_episodes_done
        print(f'Done {self.n_all_episodes} episodes')
    
    def _update_pose_buffer(self):
        # buffer the last perforemd pose in xyz locations and pop the oldest buffered frame
        body_pos = self._rigid_body_pos[:, None]
        self.pose_buffer = torch.cat([self.pose_buffer[:, 1:], body_pos], dim=1)  # [n_envs, buff_size, n_joints, 3]
        if self.save_hml_episodes:
            self.episode_buf[torch.arange(len(self.progress_buf)), self.progress_buf] = self._rigid_body_pos

    def _compute_reset(self):
        Humanoid._compute_reset(self)  # reset if fails or got to end of episode
        return
    
    def visualize(self, pose, out_path, is_prefix_comp=False, rep_type='hml', model_kwargs=None, prompts=None):
        dir_path = os.path.dirname(out_path)
        os.makedirs(dir_path, exist_ok=True)
        if rep_type == 'hml':
            pred_xyz = recover_from_ric((pose.squeeze().permute(0, 2, 1)*self.std+self.mean).float(), 22)  # [bs, seqlen, 1, 22, 3]
        elif rep_type == 'xyz':
            pred_xyz = self.rep.pose_to_hml_xyz(pose)
        else:
            raise ValueError()
        # xyz = pred_xyz.reshape(1, -1, 22, 3).detach().cpu().numpy()
        xyz = pred_xyz.squeeze().detach().cpu().numpy()
        # plot_3d.draw_to_batch(xyz, self.hml_prompts*xyz.shape[0], [out_path.replace('.mp4', f'_{i}.mp4') for i in range(xyz.shape[0])])  # Draw with T2M-GPT
        animations = np.empty(shape=(1, xyz.shape[0]), dtype=object)
        if prompts is None:
            prompts = self.hml_prompts
        for env_i in range(xyz.shape[0]):
            caption = prompts[env_i]
            motion = xyz[env_i] # .transpose(2, 0, 1)  # [:length]
            animation_save_path = out_path.replace('.mp4', f'_{env_i}.mp4')
            gt_frames = np.arange(self.context_len_20fps) if is_prefix_comp else []
            cond=None
            if model_kwargs is not None:
                cond = {k: v[env_i] for k,v in model_kwargs['y'].items() if '_cond' in k}
                # cond.update({'heading_print': heading_all[env_i]})
                if 'target_joint_names' in model_kwargs['y']:
                    cond.update({'joint_names': model_kwargs['y']['target_joint_names'][env_i]})
                if 'is_heading' in model_kwargs['y']:
                    cond.update({'is_heading': model_kwargs['y']['is_heading'][env_i]})
            animations[0, env_i] = plot_3d_motion(animation_save_path, t2m_kinematic_chain, motion, dataset='humanml', title=caption, fps=20, gt_frames=gt_frames, cond=cond)

        save_multiple_samples(out_path, {'all': 'samples_{:02d}_to_{:02d}.mp4'}, animations, 20, xyz.shape[1], no_dir=True)

        abs_path = os.path.abspath(out_path)
        print(f'[Done] Results are at [{abs_path}]')
    
    def _update_marker(self):
        if self.cfg['env']['show_markers']:
            if flags.show_traj:
                self._marker_pos[:] = self.ref_body_pos
            else:
                self._marker_pos[:] = 0
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(self._marker_actor_ids), len(self._marker_actor_ids))
        return
    
    def _zero_out_far(self, _ref_rb_pos_subset, _ref_body_vel_subset, _root_pos, _body_pos_subset, _body_vel_subset):
        close_distance = self.close_distance
        distance = torch.norm(_root_pos - _ref_rb_pos_subset[..., 0, :], dim=-1)

        zeros_subset = distance > close_distance
        _ref_rb_pos_subset[zeros_subset, 1:] = _body_pos_subset[zeros_subset, 1:]
        _ref_body_vel_subset[zeros_subset, :] = _body_vel_subset[zeros_subset, :]

        far_distance = self.far_distance  # does not seem to need this in particular...
        vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
        _ref_rb_pos_subset[vector_zero_subset, 0] = ((_ref_rb_pos_subset[vector_zero_subset, 0] - _body_pos_subset[vector_zero_subset, 0]) / distance[vector_zero_subset, None] * far_distance) + _body_pos_subset[vector_zero_subset, 0]
        return _ref_rb_pos_subset, _ref_body_vel_subset
    
    def _compute_task_obs_demo(self, env_ids=None, set_goal=None):

        # TODO - fatch from cache if possible
        motion_res = self._get_state_from_gen_cache(motion_ids=None)
        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
        motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
        motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        ref_rb_pos_orig = motion_res["rb_pos_orig"]

        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            ref_rb_pos = ref_rb_pos[env_ids]
            ref_body_vel = ref_body_vel[env_ids]

        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]

        body_pos_subset = body_pos[..., self._track_bodies_id, :]
        body_rot_subset = body_rot[..., self._track_bodies_id, :]
        body_vel_subset = body_vel[..., self._track_bodies_id, :]
        body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]


        time_steps = 1
        ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :].clone()  # clone since we use it twice in the zero_out_far case
        ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :].clone()  # clone since we use it twice in the zero_out_far case

        if self.zero_out_far:
            ref_rb_pos_subset, ref_body_vel_subset = self._zero_out_far(ref_rb_pos_subset, ref_body_vel_subset, root_pos, body_pos_subset, body_vel_subset)

        # obs are the differences between _rigid_body_pos and ref_rb_pos (sort of)
        obs = humanoid_im.compute_imitation_observations_v7(root_pos, root_rot, body_pos_subset, body_vel_subset, ref_rb_pos_subset, ref_body_vel_subset, time_steps, self._has_upright_start)
        # obs [1, 216]

        goal_obs = None
        if set_goal is not None:
            goal_ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :].clone()  # clone since we use it twice in the zero_out_far case
            goal_ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :].clone()  # clone since we use it twice in the zero_out_far case
            goal_ref_trans = goal_ref_rb_pos_subset[:, [0]]
            goal_ref_rb_pos_subset = goal_ref_rb_pos_subset - goal_ref_trans + torch.tensor(set_goal).unsqueeze(0).unsqueeze(0).to(goal_ref_rb_pos_subset.device)
            if self.zero_out_far:
                goal_ref_rb_pos_subset, goal_ref_body_vel_subset = self._zero_out_far(goal_ref_rb_pos_subset, goal_ref_body_vel_subset, root_pos, body_pos_subset, body_vel_subset)
            goal_obs = humanoid_im.compute_imitation_observations_v7(root_pos, root_rot, body_pos_subset, body_vel_subset, goal_ref_rb_pos_subset, goal_ref_body_vel_subset, time_steps, self._has_upright_start)

        # print('ref_rb_pos', ref_rb_pos[0,0])

        if len(env_ids) == self.num_envs:
            self.prev_ref_body_pos = ref_rb_pos  # not a bug - this method is called once in for each sim step at post_physics (and once again if reset occured, but then len(env_ids) != self.num_envs)
            self.ref_body_pos = ref_rb_pos
            self.ref_body_pos_subset = ref_rb_pos_orig
            self.ref_pose_aa = None

        # TODO - cache obs
        
        return obs, goal_obs
    
    
    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None, source='dataset'):
        if source == 'dataset':
            return super()._get_state_from_motionlib_cache(motion_ids, motion_times, offset)
        elif source == 'gen':
            return self._get_state_from_gen_cache(motion_ids)
        else:
            raise ValueError(f'Unsupported source [{source}]')
    
    def _get_state_from_gen_cache(self, motion_ids):

        # pose_res = requests.get(f'http://{SERVER}:8080/get_pose')
        # json_data = pose_res.json()
        json_data = self.get_pred_pose()  # xyz pred [1, 24, 3]
        ref_rb_pos = json_data["j3d"][:self.num_envs, smpl_2_mujoco]
        
        
        trans = ref_rb_pos[:, [0]]

        # if len(self.root_pos_acc) > 0 and np.linalg.norm(trans - self.root_pos_acc[-1]) > 1:
        # import ipdb; ipdb.set_trace()
        # print("juping!!")
        ref_rb_pos_orig = ref_rb_pos.clone()

        ref_rb_pos = ref_rb_pos - trans  # [bs, 24, 3]
        # ############################## Limb Length ##############################
        # limb_vecs = ref_rb_pos[:, self.limb_pairs[0]] - ref_rb_pos[:, self.limb_pairs[1]]
        # limb_lengths = torch.linalg.norm(limb_vecs, dim=-1)
        # # limb_lengths = []
        # # for i in range(6):
        # #     parent = self.parent_indices[i]
        # #     if parent != -1:
        # #         limb_lengths.append(np.linalg.norm(ref_rb_pos[:, parent] - ref_rb_pos[:, i], axis = -1))
        # # limb_lengths = np.array(limb_lengths).transpose(1, 0)
        # scale = (limb_lengths/self.mean_limb_lengths).mean(axis = -1) 
        # ref_rb_pos /= scale[:, None, None]
        # ############################## Limb Length ##############################
        # s_dt = 1/30
        
        # TODO - use actual motion in the buffes, that might avoid the jumps?
        
        # self.root_pos_acc.append(trans)
        # # filtered_root_trans = np.array(self.root_pos_acc)
        # root_trans = self.root_pos_acc.get()  # torch.cat(list(self.root_pos_acc), dim=1)  # [bs, buf_len, 3]
        # filtered_root_trans = root_trans[:, [-1], :]
        # # filtered_root_trans[..., 2] = filters.gaussian_filter1d(filtered_root_trans[..., 2], 10, axis=0, mode="mirror") # More filtering on the root translation
        # filtered_root_trans[..., [2]] = apply_one_side_filter1d_torch(root_trans[..., [2]], self.filter_kernels[10])
        # # filtered_root_trans[..., :2] = filters.gaussian_filter1d(filtered_root_trans[..., :2], 5, axis=0, mode="mirror")
        # filtered_root_trans[..., :2] = apply_one_side_filter1d_torch(root_trans[..., :2], self.filter_kernels[5])
        # trans = filtered_root_trans  # [:, [-1], :]  # [bs, 1, 3]

        # self.body_pos_acc.append(ref_rb_pos.reshape(ref_rb_pos.shape[0], 1, -1))  # [bs, buf_len, 24*3]
        # # body_pos = np.array(self.body_pos_acc)
        # filtered_ref_rb_pos = self.body_pos_acc.get()  # torch.cat(list(self.body_pos_acc), dim=1)
        # # filtered_ref_rb_pos = filters.gaussian_filter1d(body_pos, 2, axis=0, mode="mirror")
        # ref_rb_pos = apply_one_side_filter1d_torch(filtered_ref_rb_pos, self.filter_kernels[2]).reshape(ref_rb_pos.shape[0], -1, 3)
        # # ref_rb_pos = filtered_ref_rb_pos[-1]
        # # ref_rb_pos = filtered_ref_rb_pos[:, -1].reshape(ref_rb_pos.shape[0], -1, 3)  # [bs, 24, 3]

        # ref_rb_pos = torch.from_numpy(ref_rb_pos + trans).float()
        ref_rb_pos += trans
        # ref_rb_pos = ref_rb_pos.matmul(self.to_isaac_mat.T).cuda()
        ref_rb_pos = ref_rb_pos.matmul(self.to_isaac_mat.T)

        if 'j3d_prev' in json_data.keys():
            # take prev frame from humanoid insted of prediction for the window transition case
            self.prev_ref_body_pos = json_data["j3d_prev"][:self.num_envs, smpl_2_mujoco].matmul(self.to_isaac_mat.T)

        ref_body_vel = SkeletonMotion._compute_velocity(torch.stack([self.prev_ref_body_pos, ref_rb_pos], dim=1), time_delta=self.s_dt, guassian_filter=False)[:, 0]  # 

        root_pos = ref_rb_pos[..., 0, :].clone()

        empty_placeholder = torch.empty_like(root_pos)


        return {
            "root_pos": root_pos,  # TODO
            "root_rot": empty_placeholder,
            "dof_pos": empty_placeholder,
            "root_vel": empty_placeholder,
            "root_ang_vel": empty_placeholder,
            "dof_vel": empty_placeholder,
            "motion_aa": empty_placeholder,
            "rg_pos": ref_rb_pos.clone(),  # TODO  # one of the only two used
            "rb_rot": empty_placeholder,  # TODO
            "body_vel": ref_body_vel.clone(),      # one of the only two used
            "body_ang_vel": empty_placeholder,  # TODO
            "motion_bodies": empty_placeholder,
            "motion_limb_weights": empty_placeholder,
            "rb_pos_orig": ref_rb_pos_orig
        }


@torch.jit.script
def compute_imitation_reward_wo_rot(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, rwd_specs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
    w_pos, w_rot, w_vel, w_ang_vel = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"]

    # body position reward
    diff_global_body_pos = ref_body_pos - body_pos
    diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)

    # body rotation reward
    # if calc_rot:
    #     diff_global_body_rot = torch_utils.quat_mul(ref_body_rot, torch_utils.quat_conjugate(body_rot))
    #     diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
    #     diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
    #     r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)
    # else:
    r_body_rot = torch.zeros_like(r_body_pos)

    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
    r_vel = torch.exp(-k_vel * diff_global_vel_dist)

    # body angular velocity reward
    # if calc_rot:
    #     diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
    #     diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
    #     r_ang_vel = torch.exp(-k_ang_vel * diff_global_ang_vel_dist)
    # else:
    r_ang_vel = torch.zeros_like(r_body_pos)

    reward = w_pos * r_body_pos + w_rot * r_body_rot + w_vel * r_vel + w_ang_vel * r_ang_vel
    reward_raw = torch.stack([r_body_pos, r_body_rot, r_vel, r_ang_vel], dim=-1)
    # import ipdb
    # ipdb.set_trace()
    return reward, reward_raw