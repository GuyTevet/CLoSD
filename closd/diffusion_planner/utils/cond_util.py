from closd.diffusion_planner.utils.loss_util import masked_l2, angle_l2
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_root_rot_pos
import torch
import numpy as np

# STD = torch.tensor(np.load('./dataset/HumanML3D/Std.npy')).cuda()
# MEAN = torch.tensor(np.load('./dataset/HumanML3D/Mean.npy')).cuda()

def traj_cond_fn(x_zero_pred, t, y, mean, std):
    assert 'traj_cond' in y.keys() or 'heading_cond' in y.keys()
    # # DEBUG - start
    # buggy_pred = hml_to_xyz(x_zero_pred, mean, std)[:, 0, [0,2]]  # [bs, 2, n_frames]
    # # DEBUG - end
    if 'prefix' in y.keys():
        x_zero_pred = torch.cat([y['prefix'], x_zero_pred], dim=-1)
    # traj_pred =  hml_to_xyz(x_zero_pred, mean, std)[:, 0, [0,2]]  # [bs, 2, n_frames]
    heading_rot, traj_pred = hml_to_root_loc_rot(x_zero_pred, mean, std)  # [bs, 1, n_frames], # [bs, 2, n_frames]
    # # DEBUG - start
    # full_pred = traj_pred - traj_pred[..., [y['prefix'].shape[-1]]]
    # # DEBUG - end
    if 'prefix' in y.keys():
        traj_pred = traj_pred[..., y['prefix'].shape[-1]:] - traj_pred[..., [y['prefix'].shape[-1]]]
        heading_rot = heading_rot[..., y['prefix'].shape[-1]:] - heading_rot[..., [y['prefix'].shape[-1]]]

        # # DEBUG - start
        # import matplotlib.pyplot as plt
        # import os
        # save_dir = './debug'
        # os.makedirs(save_dir, exist_ok=True)
        # for sample_i in range(traj_pred.shape[0]):
        #     _traj_pred = plt.scatter(traj_pred[sample_i, 0].detach().cpu().numpy(), traj_pred[sample_i, 1].detach().cpu().numpy())
        #     _traj_cond = plt.scatter(y['traj_cond'][sample_i, 0].cpu().numpy(), y['traj_cond'][sample_i, 1].cpu().numpy())
        #     _buggy_pred = plt.scatter(buggy_pred[sample_i, 0].detach().cpu().numpy(), buggy_pred[sample_i, 1].detach().cpu().numpy())
        #     _full_pred = plt.scatter(full_pred[sample_i, 0].detach().cpu().numpy(), full_pred[sample_i, 1].detach().cpu().numpy(), marker='x', c=['black']*20+['red']*40)
        #     plt.legend((_traj_pred, _full_pred, _traj_cond, _buggy_pred),
        #                ('traj_pred', 'full_pred', 'traj_cond', 'buggy_pred'), scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
        #     plt.savefig(os.path.join(save_dir, f'sample{sample_i}_t{t[0]}.png'))
        #     plt.close()
        # # DEBUG - end

    # return masked_l2(traj_pred, y['traj_cond'], y['mask'][:, 0]).sum()

    loss = torch.tensor(0., device=traj_pred.device, dtype=traj_pred.dtype)
    if 'traj_cond' in y.keys():
        loss += masked_l2(traj_pred, y['traj_cond'], y['recon_mask']).sum()
        # print('traj loss', masked_l2(traj_pred, y['traj_cond'], y['recon_mask']))
    if 'heading_cond' in y.keys():
        loss += masked_l2(heading_rot, y['heading_cond'], y['recon_mask'], loss_fn=angle_l2).sum()
        # print('heading loss', masked_l2(heading_rot, y['heading_cond'], y['recon_mask'], loss_fn=angle_l2))

    return loss

def hml_to_xyz(sample, mean, std):
    sample = sample.permute(0, 2, 3, 1).float()
    sample = sample * std + mean
    sample = recover_from_ric(sample, joints_num=22)  # assuming humanml data
    return sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

def hml_to_root_loc_rot(sample, mean, std):
    sample = sample.permute(0, 2, 3, 1).float()
    sample = sample * std + mean
    r_rot_quat, r_pos = recover_root_rot_pos(sample)
    r_pos = r_pos[:, 0, :, [0,2]].permute(0, 2, 1)
    r_rot_ang = torch.atan2(r_rot_quat[..., 2], r_rot_quat[..., 0])  # relative to Z+, always starts with 0 (i.e. dront facing to Z+)
    return r_rot_ang, r_pos