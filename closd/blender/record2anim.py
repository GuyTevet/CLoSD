import bpy
from mathutils import Vector
import joblib
import os
import sys
sys.path.append('.')

from closd.utils.closd_util import STATES
from closd.utils.poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
import closd.blender.blender_utils as blender_utils

import argparse
import torch
import numpy as np

object_path_per_state = {
    STATES.SIT: 'closd/data/assets/fbx/sofa.fbx',
    STATES.REACH: 'closd/data/assets/fbx/reach_target.fbx',
    STATES.STRIKE_KICK: 'closd/data/assets/fbx/strike_target.fbx',
    STATES.STRIKE_PUNCH: 'closd/data/assets/fbx/strike_target.fbx',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_path", type=str, required=True, help='Input pkl file recording states')
    parser.add_argument("--results_dir", type=str, default='closd/blender/results', help='')
    parser.add_argument("--stage_path", type=str, default='closd/blender/assets/empty.blend', help='Empty blender file (possibly with predefined camera and lights).')
    parser.add_argument("--mjcf_path", type=str, default='closd/data/assets/mjcf/smpl_humanoid.xml', help='mjcf file defining the humanoid.')
    parser.add_argument("--cam_track", action='store_true', help='if true, will enable camera tracking.')
    parser.add_argument("--add_markers", action='store_true', help='if true, will add MDM moption markers.')
    parser.add_argument("--full_episods", action='store_true', help='if true, will render only full episods.')
    parser.add_argument("--single_file", action='store_true', help='if true, will put all motions in a single file.')
    parser.add_argument("--downsample_rate", type=int, default=2, help='')
    
    if "--" not in sys.argv:
        argv = []
    else:
        argv = sys.argv[sys.argv.index("--") + 1:]
    params = parser.parse_args(argv)

    orig_framerate = 60  # fps
    new_framerate = int(orig_framerate / params.downsample_rate)
    record_data = joblib.load(params.record_path)

    all_n_frames = [_data['ref_body_pos_full'][0::params.downsample_rate].shape[0] for _data in record_data.values()]
    full_episod_len = max(all_n_frames)
    print(f'Full episod length is [{full_episod_len}]')

    all_path = os.path.join(params.results_dir, os.path.basename(params.record_path).replace('.pkl', ''), 'all.blend')
    if params.single_file:
        bpy.ops.wm.open_mainfile(filepath=params.stage_path)

    for sample_i, (sample_name, sample_data) in enumerate(record_data.items()):

        env_id, try_id = [int(s) for s in sample_name.split('_')]
        out_path = os.path.join(params.results_dir, os.path.basename(params.record_path).replace('.pkl', ''), sample_name + '.blend')
        markers_data = sample_data['ref_body_pos_full'][0::params.downsample_rate]
        n_frames, n_joints, _ = markers_data.shape
        
        if params.full_episods and n_frames < full_episod_len:
            print(f'Skipping [{sample_name}]')
            continue

        pose_quat, trans = sample_data['body_quat'].numpy()[0::params.downsample_rate], sample_data['trans'].numpy()[0::params.downsample_rate]
        skeleton_tree = SkeletonTree.from_dict(sample_data['skeleton_tree'])
        offset = skeleton_tree.local_translation[0]
        root_trans_offset = trans - offset.numpy()
        sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat), torch.from_numpy(trans), is_local=True)
        quat_data = sk_state.global_rotation
        pos_data = sk_state.global_translation.cpu().numpy()
        euler_data = blender_utils.quaternion_to_euler_angle_vectorized1(quat_data.reshape(n_frames*n_joints, 4)).reshape(n_frames, n_joints, 3)
        
        # smooth_end_effectors:
        euler_data = blender_utils.smooth_end_effectors(euler_data, window_size=10)
        pos_data = blender_utils.smooth_end_effectors(pos_data)


        print(f'Visualizing sample [{sample_name}] with [{n_frames}] frames.')
        if not params.single_file:
            bpy.ops.wm.open_mainfile(filepath=params.stage_path)
        blender_utils.load_humanoid(params.mjcf_path, motion_name=sample_name)
        if params.add_markers:
            blender_utils.add_full_body_markers(sample_name)
        scn = bpy.context.scene
        scn.frame_start = 1
        scn.frame_end = n_frames
        scn.render.fps = new_framerate
        for joint_i, joint_name in enumerate(blender_utils.mujoco_joint_names):
            _obj = blender_utils.get_object_by_idx(sample_name, joint_i)
            for frame_i in range(n_frames):
                # Set location-rotation as a keyframe
                _pos = pos_data[frame_i, joint_i]
                _rot = euler_data[frame_i, joint_i]
                _obj.location = Vector(_pos)
                _obj.rotation_euler = Vector(_rot)
                _obj.keyframe_insert('location', frame=frame_i+1)
                _obj.keyframe_insert('rotation_euler', frame=frame_i+1)
                if params.add_markers:
                    _marker = bpy.data.objects[sample_name + '_marker_' + joint_name]
                    _marker_pos = markers_data[frame_i, joint_i]
                    _marker.location = Vector(_marker_pos)
                    _marker.keyframe_insert('location', frame=frame_i+1)
        
        # add target object
        has_target = 'target_trans' in sample_data.keys() and 'init_state' in sample_data.keys()
        if has_target:
            target_coll = blender_utils.create_collection(sample_name + "_target")
            default_collection = bpy.data.collections["Collection"]
            
            if len(sample_data['target_trans'].shape) == 2:
                target_path = object_path_per_state[int(sample_data['init_state'].cpu().numpy())]
                bpy.ops.import_scene.fbx(filepath=target_path)
                target_trans = sample_data['target_trans'].cpu().numpy()[0::params.downsample_rate]  # [frames, 3]
                target_quat = sample_data['target_quat'].cpu().numpy()[0::params.downsample_rate]  # [frames, 4]
                target_euler = blender_utils.quaternion_to_euler_angle_vectorized1(target_quat)  # [frames, 3]
                if 'sofa' in target_path:
                    target_euler[:, -1] -= np.pi/2
                target_obj = bpy.data.objects['target']
                target_obj.name = sample_name + '_' + target_obj.name
                blender_utils.move_collection(target_obj, default_collection, target_coll)
                for frame_i in range(1, n_frames):
                    target_obj.location = Vector(target_trans[frame_i])
                    target_obj.rotation_euler = Vector(target_euler[frame_i])
                    target_obj.keyframe_insert('location', frame=frame_i+1)
                    target_obj.keyframe_insert('rotation_euler', frame=frame_i+1)
            if len(sample_data['target_trans'].shape) == 3:
                target_path = [object_path_per_state[int(STATES.REACH)], object_path_per_state[int(STATES.STRIKE_PUNCH)], object_path_per_state[int(STATES.SIT)]]
                obj_names = ['reach_target', 'strike_target', 'sit_target']
                for p, name in zip(target_path, obj_names):
                    bpy.ops.import_scene.fbx(filepath=p)
                    bpy.data.objects['target'].name = name
                target_trans = sample_data['target_trans'].cpu().numpy()[0::params.downsample_rate]  # [frames, 3]
                target_quat = sample_data['target_quat'].cpu().numpy()[0::params.downsample_rate]  # [frames, 4]
                target_euler = blender_utils.quaternion_to_euler_angle_vectorized1(target_quat)  # [frames, 3]
                target_euler[..., -1, -1] -= np.pi/2  # sofa case
                for obj_i, name in enumerate(obj_names):
                    target_obj = bpy.data.objects[name]
                    target_obj.name = sample_name + '_' + target_obj.name
                    blender_utils.move_collection(target_obj, default_collection, target_coll)
                    for frame_i in range(1, n_frames):
                        target_obj.location = Vector(target_trans[frame_i, obj_i])
                        target_obj.rotation_euler = Vector(target_euler[frame_i, obj_i])
                        target_obj.keyframe_insert('location', frame=frame_i+1)
                        target_obj.keyframe_insert('rotation_euler', frame=frame_i+1)

        
        # Camera tracking
        if params.cam_track:
            camera = bpy.data.objects['Camera']
            for frame_i in range(1, n_frames):
                delta = pos_data[frame_i, 0] - pos_data[frame_i-1, 0]
                camera.location.x += delta[0]
                camera.location.y += delta[1]
                # camera.location.z += delta[2]
                camera.keyframe_insert('location', frame=frame_i+1)
        
        # Saving result
        if not params.single_file:
            blender_utils.remove_collection("Collection")
            blender_utils.save_blend(out_path)
    
    if params.single_file:
        blender_utils.remove_collection("Collection")
        blender_utils.save_blend(all_path)
