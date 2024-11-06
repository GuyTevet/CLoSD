import torch
import pickle
from closd.diffusion_planner.data_loaders.humanml.networks.modules import *
from closd.diffusion_planner.data_loaders.humanml.networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
from closd.diffusion_planner.utils import dist_util
from closd.diffusion_planner.data_loaders import humanml_utils
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import get_target_location, sample_goal
from closd.diffusion_planner.data_loaders.humanml.utils.metrics import calculate_skating_ratio, calculate_mean_penetration, calculate_penetration, calculate_floating, calculate_foot_sliding
from closd.diffusion_planner.data_loaders.humanml.utils.utils import sample_to_motion
from closd.diffusion_planner.utils.sampler_util import AutoRegressiveSampler
from closd.diffusion_planner.utils.loss_util import masked_goal_l2


def build_models(opt):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
                                        pos_size=opt.dim_pos_ohot,
                                        hidden_size=opt.dim_text_hidden,
                                        device=opt.device)
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception("Text Encoder Mode not Recognized!!!")

    seq_prior = TextDecoder(text_size=text_size,
                            input_size=opt.dim_att_vec + opt.dim_movement_latent,
                            output_size=opt.dim_z,
                            hidden_size=opt.dim_pri_hidden,
                            n_layers=opt.n_layers_pri)


    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    len_estimator = MotionLenEstimatorBiGRU(opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes)

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)
    checkpoints = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_est_bigru', 'model', 'latest.tar'), map_location=opt.device)
    len_estimator.load_state_dict(checkpoints['estimator'])
    len_estimator.to(opt.device)
    len_estimator.eval()

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, len_estimator

class CompV6GeneratedDataset(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt)
        trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
        epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 10 if opt.dataset_name == 't2m' else 6
        # print(mm_idxs)

        print('Loading model: Epoch %03d Schedule_len %03d' % (epoch, schedule_len))
        trainer.eval_mode()
        trainer.to(opt.device)
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                tokens = tokens[0].split('_')
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
                pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False

                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)

                    m_lens = mov_length * opt.unit_length
                    pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens,
                                                          m_lens[0]//opt.unit_length, opt.dim_pose)
                    if t == 0:
                        # print(m_lens)
                        # print(text_data)
                        sub_dict = {'motion': pred_motions[0].cpu().numpy(),
                                    'length': m_lens[0].item(),
                                    'cap_len': cap_lens[0].item(),
                                    'caption': caption[0],
                                    'tokens': tokens}
                        generated_motion.append(sub_dict)

                    if is_mm:
                        mm_motions.append({
                            'motion': pred_motions[0].cpu().numpy(),
                            'length': m_lens[0].item()
                        })
                if is_mm:
                    mm_generated_motions.append({'caption': caption[0],
                                                 'tokens': tokens,
                                                 'cap_len': cap_lens[0].item(),
                                                 'mm_motions': mm_motions})

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.opt = opt
        self.w_vectorizer = w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.opt.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompMDMGeneratedDataset(Dataset):

    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1., hml_type=None):
        self.args = args
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.model = model
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
        if self.args.autoregressive:
            sample_cls = AutoRegressiveSampler(args, sample_fn)
            sample_fn = sample_cls.sample

        self.hml_type = hml_type
        if self.hml_type is not None:
            _t = lambda x: torch.tensor(x, device=dist_util.dev())[None, :, None, None]
            self.used_mean = _t(np.load(pjoin(self.dataset.opt.data_root, f'Mean_{self.hml_type}.npy')))
            self.used_std = _t(np.load(pjoin(self.dataset.opt.data_root, f'Std_{self.hml_type}.npy')))
            self.orig_mean = _t(np.load(pjoin(self.dataset.opt.data_root, f'Mean.npy')))
            self.orig_std = _t(np.load(pjoin(self.dataset.opt.data_root, f'Std.npy')))

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()

        self.aux_metrics = []

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}
                motion = motion.to(dist_util.dev())

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                init_image = None
                if args.spatial_condition is not None:
                    if args.spatial_condition == 'traj':
                        model_kwargs['y']['condition_mask'] = torch.tensor(humanml_utils.HML_ROOT_HORIZONTAL_MASK[None, :, None, None], device=dist_util.dev())
                        init_image = torch.tensor(motion * model_kwargs['y']['condition_mask'], device=dist_util.dev())
                        model_kwargs['y']['condition_input'] = init_image.clone()

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale
                
                if self.args.multi_target_cond:
                    assert self.args.target_joint_names == 'DIMP_FINAL', 'We dont support the rest of the configurations anymore'
                    # The reason is that we evaluate PURE_T2M (see below), and DIMP_FINAL is the only one supporting it!

                    # model_kwargs['y']['target_joint_names'], model_kwargs['y']['is_heading'] = sample_goal(dataloader.batch_size, motion.device, self.args.target_joint_names)
                    model_kwargs['y']['target_joint_names'], model_kwargs['y']['is_heading'] = sample_goal(dataloader.batch_size, motion.device, 'PURE_T2M')
                    model_kwargs['y']['target_cond'] = get_target_location(motion, 
                                    self.dataset.mean_gpu, self.dataset.std_gpu, 
                                    torch.tensor([motion.shape[-1]] * motion.shape[0]), self.dataset.t2m_dataset.opt.joints_num, 
                                    model.all_goal_joint_names, model_kwargs['y']['target_joint_names'], model_kwargs['y']['is_heading'])

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=init_image,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )

                    if self.hml_type == 'global_root':
                        sample = self.convert_global_root(sample)

                    # update length for the prefix case
                    if 'prefix' in model_kwargs['y'].keys():
                        model_kwargs['y']['lengths'][:] = sample.shape[-1]
                    
                    # if self.args.multi_target_cond:
                    #     cur_metrics = self.calc_target_metrics(sample[..., -self.args.pred_len:], model_kwargs['y'])
                    #     self.aux_metrics.append(cur_metrics)

                    cur_metrics = self.physical_metrics(sample, model_kwargs['y']['lengths'])
                    self.aux_metrics.append(cur_metrics)

                    if t == 0:
                        sub_dicts = [{
                            'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                            'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                            'caption': model_kwargs['y']['text'][bs_i],
                            'tokens': tokens[bs_i],
                            # Fixed cap_len calculation, changed from len(tokens[bs_i])
                            # Lead to improved R-precision and Multimodal Dist.
                            # issue: https://github.com/GuyTevet/motion-diffusion-model/issues/182
                            'cap_len': tokens[bs_i].index('eos/OTHER') + 1, 
                            } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def convert_global_root(self, sample):
        # normed global_root (used) -> normed relative_root (orig)
        denormed_global = sample * self.used_std + self.used_mean
        denormed_global[:, :3, :, :] -= denormed_global[:, :3, :, [0]]  # Center the first frame
        denormed_relative = global_to_relative(denormed_global)
        # assert (relative_to_global(global_to_relative(denormed_global)) - denormed_global < 1e-6).all()  # Convertion test
        normed_relative = (denormed_relative - self.orig_mean) / self.orig_std
        return normed_relative


    def calc_target_metrics(self, pred_motion, cond):
        pred_location = get_target_location(pred_motion, self.dataset.mean_gpu, self.dataset.std_gpu,
                                            torch.tensor([pred_motion.shape[-1]] * pred_motion.shape[0]), 
                                            self.dataset.t2m_dataset.opt.joints_num, self.model.all_goal_joint_names,
                                            cond['target_joint_names'], cond['is_heading'])
        return {
            'target_dist': masked_goal_l2(pred_location, cond['target_cond'], cond, self.model.all_goal_joint_names).mean()
        } 

    def physical_metrics(self, pred_motion, lengths):
        pred_motion = sample_to_motion(pred_motion, self.dataset, self.model)
        skate_ratio, skate_vel = calculate_skating_ratio(pred_motion)
        mean_penetration = calculate_mean_penetration(pred_motion)
        penetration = calculate_penetration(pred_motion, lengths)
        floating = calculate_floating(pred_motion, lengths)
        skating = calculate_foot_sliding(pred_motion, lengths)
        return {'skate_ratio': torch.tensor(skate_ratio), 'mean_penetration': torch.tensor(mean_penetration), 
                'penetration': torch.tensor(penetration), 'floating': torch.tensor(floating), 'skating': torch.tensor(skating)} #TODO: Remove dependence on torch, use numpy instead

    
    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        if self.dataset.mode == 'eval':
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
    

def global_to_relative(data):
    """ Convert global root rotation and orientation of motion data to relative.

    Args:
        data (torch.Tensor): Motion data in shape B x n_joints x n_features x n_frames

    Returns:
        torch.Tensor: Motion data where root data is transformed to relative - same shape as input
    """
    output = data.clone()
    rel_rot, rel_pos = undo_recover_root_rot_pos(output.permute(0, 2, 3, 1))
    output[:, :1] = rel_rot.permute(0, 3, 1, 2)
    output[:, 1:4] = rel_pos.permute(0, 3, 1, 2)[:, [0, 2, 1]]
    return output

def undo_recover_root_rot_pos(data):
    from closd.diffusion_planner.data_loaders.humanml.common.quaternion import qrot
    gl_pos = data[..., 1:4][...,[0, 2, 1]]
    gl_rot = data[..., :1]
    rel_pos = torch.zeros_like(gl_pos).to(data.device)
    rel_pos[:, :, 1:, [0, 2]] = gl_pos[:, :, 1:, [0, 2]] - gl_pos[:, :, :-1, [0, 2]] 
    gl_quat_rot = torch.zeros(gl_rot.shape[:-1] + (4,)).to(data.device)
    gl_quat_rot[..., :1] = torch.cos(gl_rot)
    gl_quat_rot[..., 2:3] = torch.sin(gl_rot)
    rel_pos = qrot(gl_quat_rot, rel_pos) # rel_pos[:,:,0] is 0 now
    rel_pos[:,:,:-1] = rel_pos[:,:,1:].clone() # very last element of relative positions is lost and the first is not necessarily 0 # try setting it to 0 too
    #rel_pos[:,:,-1] = torch.zeros_like(rel_pos[:,:,-1])
    rel_pos[..., 1] = data[..., 3]
    rel_rot = torch.zeros_like(gl_rot).to(data.device)
    rel_rot[:, :, :-1, :] = gl_rot[:, :, 1:, :] - gl_rot[:, :, :-1, :]
    return rel_rot, rel_pos


class CompMDMExternalDataset(CompMDMGeneratedDataset):

    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1., hml_type=None):
        self.args = args
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.model = model
        assert mm_num_samples < len(dataloader.dataset)
        self.max_motion_length = max_motion_length
        self.hml_type = hml_type

        # Load motion from files
        if args.external_results_file.endswith('.npy'):
            external_motions = np.load(args.external_results_file, allow_pickle=True).item()  
        else:
            with open(args.external_results_file, 'rb') as f:
                external_motions = pickle.load(f)
                
        # make the dataset contain unique keys only, i.e., one equivalent for each motin from the GT
        if args.do_unique and 'db_key' in external_motions:
            
            # shuffle file items, so in each iteration we will get a unique key from a different place in the input file
            # this is important because the same key can be generated with multple prompts, and we would like to sample them all
            shuffled_idx = np.random.permutation(range(external_motions['motion'].shape[0]))
            print(f'new shuffled order: {shuffled_idx[:5]}')
            print(f'keys before shuffle: {external_motions["db_key"][:5]}')
            for k, v in external_motions.items():
                if isinstance(v, list):
                    external_motions[k] = np.array(v)[shuffled_idx].tolist()
                else:
                    external_motions[k] = v[shuffled_idx]
            print(f'keys after shuffle: {external_motions["db_key"][:5]}')
            
            # unique-ify
            _, unique_idx = np.unique(external_motions['db_key'], return_index=True)
            unique_idx.sort()
            for k, v in external_motions.items():
                if isinstance(v, list):
                    external_motions[k] = [v[i] for i in unique_idx]
                else:
                    external_motions[k] = v[unique_idx]
            assert all([len(external_motions[k]) == len(unique_idx) for k in external_motions.keys()]), 'Unique motion count mismatch'
            print(f'no. unique keys: {len(unique_idx)}')
            
        # FIXME: the following is commented out because it was added in a later stage. should be uncommented after submission to ICLR    
        # if num_samples_limit is not None:
        #     for k, v in external_motions.items():
        #         external_motions[k] = v[:num_samples_limit]
        
        # handle cropped motions:
        # 1) if generated tensors are shorter than max possible length
        # 2) if generated tensors are of max length but the motion is shorter, e.g., if generated by CLoSD with save_failed_episodes=True
        #    in this case, after the end of the motion there are zeros or nans 
        n_motions, max_length, _ = external_motions['motion'].shape
        adjusted_length = [max_length] * n_motions
        non_zero_length = [max_length] * n_motions
        for i, motion in enumerate(external_motions['motion']):
            motion[torch.isnan(motion)] = 0
            non_zero_length[i] = torch.nonzero(motion.sum(axis=1))[-1].item()+1 if len(torch.nonzero(motion.sum(axis=1))) > 0 else 0
            adjusted_length[i] = min(max_length, non_zero_length[i], external_motions['length'][i])
        # adjusted_length = [m if m <= max_length else max_length for m in external_motions['length']]
        idx_non_zero = np.nonzero(np.array(non_zero_length) < max_length)[0]
        if idx_non_zero.any():
            print(f'truncated length for texts: \n {[(adjusted_length[i], external_motions["caption"][i]) for i in idx_non_zero]}')
        else:
            print('ftruncated length for texts: None')
        self.generated_motion = [
            {'motion': motion, 'length': length, 'caption': caption, 'tokens': tokens.split('_'), 'cap_len': tokens.split('_').index('eos/OTHER')+1} 
             for motion, length, caption, tokens in 
             zip (external_motions['motion'], adjusted_length, external_motions['caption'], external_motions['tokens'])
            ]
        # for motion in self.generated_motion:
        #     motion.update({'cap_len': motion['tokens'].index('eos/OTHER')})
        #     motion['tokens'] = process_tokens(motion['tokens'].split(' '))
        #     # motion['motion']= motion['motion'].squeeze().permute(1, 0)
        self.mm_generated_motion = []
        self.w_vectorizer = dataloader.dataset.w_vectorizer

        self.aux_metrics = []

        if True:#self.args.physical_metrics:
            # Prepare motions and lengths for physical metric computation:
            motions_list = [gen_m['motion'][None] for gen_m in self.generated_motion]
            all_motions = torch.cat(motions_list).permute(0, 2, 1)[:,:,None,:] # B x 263 x 1 x n_frames 
            lengths_list = [np.array(int(gen_m['length']))[None] for gen_m in self.generated_motion]
            all_lengths = torch.tensor(np.concatenate(lengths_list)) # B
            
            cur_metrics = self.physical_metrics(all_motions, all_lengths)
            self.aux_metrics.append(cur_metrics)
