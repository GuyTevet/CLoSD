# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from closd.diffusion_planner.utils.fixseed import fixseed
import os
import numpy as np
import torch
from closd.diffusion_planner.utils.parser_util import generate_args
from closd.diffusion_planner.utils.model_util import create_model_and_diffusion, load_saved_model
from closd.diffusion_planner.utils import dist_util
from closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel, AutoRegressiveSampler
from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import recover_from_ric, traj_global2vel, recover_root_rot_pos, get_target_location, recover_root_rot_heading_ang, sample_goal
import closd.diffusion_planner.data_loaders.humanml.utils.paramUtil as paramUtil
from closd.diffusion_planner.data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from closd.diffusion_planner.data_loaders.tensors import collate
from moviepy.editor import clips_array
from closd.diffusion_planner.utils import cond_util
from closd.diffusion_planner.data_loaders import humanml_utils


def main(args=None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    n_joints = 22 if args.dataset == 'humanml' else 21
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps)) if args.motion_length is not None else max_frames
    cfg_type = args.__dict__.get('cfg_type', 'text')
    if args.pred_len > 0 and not args.autoregressive:
        n_frames = args.pred_len
        max_frames = args.context_len + args.pred_len
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
        
        if args.guidance_param != 1.:
            out_path += '_CFG{}{}'.format(cfg_type, args.guidance_param)
        
        if args.sampling_mode in ['goal', 'traj']:
            out_path += f'_{args.sampling_mode}'
        if args.use_recon_guidance:
            out_path += f'_recon{args.recon_param}-step{args.recon_step_start}-{args.recon_step_stop}-frame{args.recon_frame_start}-{args.recon_frame_stop}'
        
        if args.multi_target_cond:
            out_path += f'_source-{args.target_joint_source}'
            if args.target_joint_names is not None:
                out_path += '_joints-{}'.format(args.target_joint_names.replace(',', '-'))

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    sample_fn = diffusion.p_sample_loop
    if args.autoregressive:
        sample_cls = AutoRegressiveSampler(args, sample_fn, n_frames)
        sample_fn = sample_cls.sample

    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model, cfg_type)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    motion_shape = (args.batch_size, model.njoints, model.nfeats, n_frames)

    if is_using_data:
        iterator = iter(data)
        input_motion, model_kwargs = next(iterator)
        input_motion = input_motion.to(dist_util.dev())
        if 'prefix' in model_kwargs['y'].keys():
            input_motion_w_prefix = torch.concat([model_kwargs['y']['prefix'].to(dist_util.dev()), input_motion], dim=3); 
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        else:
            # a2m
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                            arg, one_action, one_action_text in zip(collate_args, action, action_text)]
        _, model_kwargs = collate(collate_args)

    model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}
    init_image = None
    if args.spatial_condition is not None:
        if args.spatial_condition == 'traj':
            model_kwargs['y']['condition_mask'] = torch.tensor(humanml_utils.HML_ROOT_HORIZONTAL_MASK[None, :, None, None])
            init_image = torch.tensor(input_motion * model_kwargs['y']['condition_mask'], device=dist_util.dev())
            model_kwargs['y']['condition_input'] = init_image.clone()  # FIXME - GUY - Looks like a bug, why using the traj from data if we override it with a stright line?
        else:
            raise ValueError(f'unsupported spatial_condition [{args.spatial_condition}]') 
    
    
    # For guidance experiments
    cond_fn = None
 
    if args.sampling_mode == 'goal':
        target_joint_names, is_heading = sample_goal(args.num_samples, dist_util.dev(), args.target_joint_names)
        target_loc = get_target_location(input_motion, data.dataset.mean_gpu, data.dataset.std_gpu, 
                                            model_kwargs['y']['lengths'], data.dataset.t2m_dataset.opt.joints_num, model.all_goal_joint_names, target_joint_names, is_heading)
        if args.target_joint_source == 'random':
            # target location    
            target_loc = torch.zeros_like(target_loc)
            # is_foot, is_pelvis, is_wrist, is_traj, is_head = [is_substr_in_list(organ, target_joint_names) for organ in ['foot', 'pelvis', 'wrist', 'traj', 'head']]
            mid_point_height = { 'traj': 0., 'pelvis': 0.93, 'right_wrist': 0.93, 'left_wrist': 0.93, 'head': 1.7, 'left_foot': 0.3, 'right_foot': 0.3}
            
            all_joint_names = model.all_goal_joint_names + ['traj', 'heading']
            for sample_i, joint_list in enumerate(target_joint_names):
                if is_heading[sample_i]:
                    target_loc[sample_i, all_joint_names.index('heading'), 0] = (2 * torch.pi * torch.rand([1], dtype=target_loc.dtype, device=target_loc.device) - torch.pi)
                for j in joint_list:
                    mid_point = torch.zeros([3], dtype=target_loc.dtype, device=target_loc.device)
                    mid_point[1] = mid_point_height[j]
                    radius = torch.ones([3], dtype=target_loc.dtype, device=target_loc.device)
                    radius[1] = 0.3
                    target_loc[sample_i, all_joint_names.index(j)] = mid_point + 2 * radius * torch.rand_like(mid_point) - radius
            target_loc[:, -2, 1] = 0   # zero the y axis for the trajectory

        model_kwargs['y']['target_cond'] = target_loc
        model_kwargs['y']['target_joint_names'] = target_joint_names
        model_kwargs['y']['is_heading'] = is_heading
            
    if args.use_inpainting:
        raise NotImplementedError()
    
    if args.spatial_condition is not None:
        raise NotImplementedError()

    
    all_motions = []
    all_lengths = []
    all_text = []
    all_target_dist = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        
        if 'text' in model_kwargs['y'].keys():
            # encoding once instead of each iteration saves lots of time
            model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])

        sample = sample_fn(
            model,
            motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=init_image,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            recon_guidance=args.use_recon_guidance,
            cond_fn=cond_fn,
        )
        

        if args.multi_target_cond:
            prefix_end_heading = get_target_location(sample[..., :args.context_len], data.dataset.mean_gpu, data.dataset.std_gpu, 
                                                    torch.tensor([args.context_len] * args.num_samples), 
                                                    data.dataset.t2m_dataset.opt.joints_num, model.all_goal_joint_names,
                                                    model_kwargs['y']['target_joint_names'], is_heading=model_kwargs['y']['is_heading'])[:, -1, 0]
            model_kwargs['y']['heading_cond'] = prefix_end_heading + model_kwargs['y']['target_cond'][:, -1, 0]
            model_kwargs['y']['heading_pred_cond'] = get_target_location(sample, data.dataset.mean_gpu, data.dataset.std_gpu, 
                                                    torch.tensor([sample.shape[-1]] * args.num_samples), 
                                                    data.dataset.t2m_dataset.opt.joints_num, model.all_goal_joint_names,
                                                    model_kwargs['y']['target_joint_names'], is_heading=model_kwargs['y']['is_heading'])[:, -1, 0]
     
        
        if model.data_rep == 'hml_vec':
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints, args.hml_type,)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        heading_all = recover_root_rot_heading_ang(sample)
        heading_all *= 180 / torch.pi
        heading_all = heading_all.cpu().numpy().round().squeeze()

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

        all_motions.append(sample.cpu().numpy())
        _len = model_kwargs['y']['lengths'].cpu().numpy()
        if 'prefix' in model_kwargs['y'].keys():
            _len += model_kwargs['y']['prefix'].shape[-1]  # assuming a fixed len prefix
        all_lengths.append(_len)

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)
    max_vis_samples = 6
    num_vis_samples = min(args.num_samples, max_vis_samples)
    animations = np.empty(shape=(args.num_samples, args.num_repetitions), dtype=object)

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            # length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)  # [:length]
            # if motion.shape[0] > length:
            #     motion[length:-1] = motion[length-1]  # duplicate the last frame to end of motion, so all motions will be in equal length
            save_file = sample_file_template.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            gt_frames = np.arange(args.context_len) if args.context_len > 0 and not args.autoregressive else []
            cond = {k: v[sample_i].cpu() for k,v in model_kwargs['y'].items() if '_cond' in k}
            cond.update({'heading_print': heading_all[sample_i]})
            if 'target_joint_names' in model_kwargs['y'].keys():
                cond.update({'joint_names': model_kwargs['y']['target_joint_names'][sample_i]})
                cond.update({'is_heading': model_kwargs['y']['is_heading'][sample_i]})
            animations[sample_i, rep_i] = plot_3d_motion(animation_save_path, 
                                                         skeleton, motion, dataset=args.dataset, title=caption, 
                                                         fps=fps, gt_frames=gt_frames, cond=cond)
            rep_files.append(animation_save_path)

    save_multiple_samples(out_path, {'all': all_file_template}, animations, fps, max(list(all_lengths) + [n_frames]))

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')

    return out_path


def save_multiple_samples(out_path, file_templates,  animations, fps, max_frames, no_dir=False):
    
    num_samples_in_out_file = 3
    n_samples = animations.shape[0]
    
    for sample_i in range(0,n_samples,num_samples_in_out_file):
        last_sample_i = min(sample_i+num_samples_in_out_file, n_samples)
        all_sample_save_file = file_templates['all'].format(sample_i, last_sample_i-1)
        if no_dir and n_samples <= num_samples_in_out_file:
            all_sample_save_path = out_path
        else:
            all_sample_save_path = os.path.join(out_path, all_sample_save_file)
            print(f'saving {os.path.split(out_path)[1]}/{all_sample_save_file}')

        clips = clips_array(animations[sample_i:last_sample_i])
        clips.duration = max_frames/fps
        
        # import time
        # start = time.time()
        clips.write_videofile(all_sample_save_path, fps=fps, threads=4, logger=None)
        # print(f'duration = {time.time()-start}')
        
        for clip in clips.clips: 
            # close internal clips. Does nothing but better use in case one day it will do something
            clip.close()
        clips.close()  # important
 

def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='train' if args.pred_len > 0 else 'text_only',
                              hml_type=args.hml_type,
                              fixed_len=args.pred_len + args.context_len, pred_len=args.pred_len, device=dist_util.dev())
    data.fixed_length = n_frames
    return data


def is_substr_in_list(substr, list_of_strs):
    return np.char.find(list_of_strs, substr) != -1  # [substr in string for string in list_of_strs]

if __name__ == "__main__":
    main()
