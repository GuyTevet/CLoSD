import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[], cond=None):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        # fig.suptitle(title, fontsize=10)
        ax.grid(b=None)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
    
    def add_traj(x_offset, z_offset, plane_height):
        ax.scatter3D(trajec[:, 0]+x_offset, 
                    plane_height, 
                    trajec[:, 1]+z_offset,
                    s=5,c=['b' if i in gt_frames else 'orange' for i in range(trajec[:, 0].shape[0])])


    def add_cond(x_offset, z_offset, plane_height):
        if cond is not None:
            start = trajec[len(gt_frames)]  # FIXME - assuming prefix completion
            for k in ['goal_cond', 'heading_cond', 'heading_pred_cond']:
                if k in cond.keys():
                    _s = 50 if k == 'goal_cond' else 5
                    if 'heading' in k and cond['is_heading']: # heading angle
                        if cond['is_heading']:  # if heading is not tested, do not draw it
                            c = 'b' if k == 'heading_cond' else 'r'
                            scale = 0.4
                            ax.quiver(0,0,0,scale*np.sin(cond[k]),0,scale*np.cos(cond[k]), color=c)
                    else:
                        for idx, goal_cond in enumerate(cond['goal_cond']):
                            if is_joint_cond[idx]:  # todo: do not rely on zero values
                                _x, _y, _z = goal_cond
                                c = cond_colors[idx % len(cond_colors)]
                                ax.scatter3D(start[0]+_x+x_offset, _y + plane_height, start[1]+_z+z_offset, s=_s,c=c)


    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    if cond is not None and 'target_cond' in cond.keys():
        is_joint_cond = (cond['target_cond']!=0).any(dim=1)
        # sanity check: pelvis should be identical if pelvis location is conditioned
        # assert not is_joint_cond[0] or np.abs(cond['pred_target_cond'][0].cpu().numpy() - data[-1,0]).max().round(8) < 1e-5
        # if is_joint_cond[0] and np.abs(cond['pred_target_cond'][0].cpu().numpy() - data[-1,0]).max() > 1e-5:
        #     print(f"diff between pred_target_cond[0] and data[-1,0] is {np.abs(cond['pred_target_cond'][0].cpu().numpy() - data[-1,0]).max()}")
        #     print(f"pred_target_cond[0]: {cond['pred_target_cond'][0].cpu().numpy().round(3)}")            
        #     print(f"pdata[-1,0]: {data[-1,0].round(3)}")
        
    # preparation related to specific datasets
    if dataset == 'kit':
        scale = 0.003  # scale for visualization
    elif dataset == 'humanml':
        scale = 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        scale = -1.5 # reverse axes, scale for visualization
    data *= scale

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue
    
    n_frames = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]  # memorize original x,z pelvis values

    # locate x,z pelvis values of ** each frame ** at zero
    data[..., 0] -= data[:, 0:1, 0] 
    data[..., 2] -= data[:, 0:1, 2]

    if cond is not None and 'target_cond' in cond.keys():
        # condition should go trhough the same transformations as the data
        # for key in ['target_cond', 'pred_target_cond']:
        for key in ['target_cond']:
            cond[key] = cond[key].cpu().numpy()
            cond[key][:-1] *= scale  # multiply non-heading values by scale
            cond_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            cond['goal_cond'] = cond['target_cond'][:-1] #[is_cond]
            # cond['heading_cond'] = cond['target_cond'][-1, 0]
            # cond['heading_pred_cond'] = cond['pred_target_cond'][-1, 0]


    def update(index):
        # sometimes index is equal to n_frames/fps due to floating point issues. in such case, we duplicate the last frame
        index = min(n_frames-1, int(index*fps))
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        _title = title + f' [{index}]'
        if cond is not None and 'heading_print' in cond.keys():
            _title += ' [{}]'.format(cond['heading_print'][index])
        if cond is not None and 'joint_names' in cond.keys():
            _title += ' {}'.format(cond['joint_names'])
        fig.suptitle(_title, fontsize=10)
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        add_cond(-trajec[index, 0], -trajec[index, 1], height_offset)
        add_traj(-trajec[index, 0], -trajec[index, 1], height_offset)

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)

        plt.axis('off')
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


        return mplfig_to_npimage(fig)

    ani = VideoClip(update)
    
    plt.close()
    return ani