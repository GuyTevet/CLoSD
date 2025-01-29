import bpy
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from closd.blender.obj_utils import xml2mesh
import sys
sys.path.append('.')
import os

mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
end_effector_names = ['R_Toe', 'L_Toe', 'R_Hand', 'L_Hand']

def move_collection(obj, src_col, trg_col):
    # recursive
    src_col.objects.unlink(obj)
    trg_col.objects.link(obj)
    for child in obj.children:
        move_collection(child, src_col, trg_col)

def create_collection(col_name):
    col = bpy.data.collections.new(col_name)
    bpy.context.scene.collection.children.link(col)
    return col

def remove_collection(col_name):
    default_collection = bpy.data.collections[col_name]
    bpy.data.collections.remove(default_collection)


def save_blend(out_path):
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f'Saved to [{out_path}]')

def state2mat(pos, rot):
    Rm = sRot.from_quat(rot)
    matrix_l = np.hstack((Rm.as_matrix(), np.mat(pos).T))
    matrix_l = np.vstack((matrix_l, np.mat([0, 0, 0, 1])))
    return matrix_l.A

def load_humanoid(mjcf_path, assets_dir='closd/blender/assets', motion_name='motion'):
    assert mjcf_path.endswith('.xml')
    mjcf_path_name = os.path.basename(mjcf_path).replace('.xml', '')
    obj_dir_path = os.path.join(assets_dir, mjcf_path_name)
    if os.path.exists(obj_dir_path):
        print(f'Found the obj dir [{obj_dir_path}].')
    else:
        print(f'Didnt found the obj dir [{obj_dir_path}] -> hence creating it!')
        xml2mesh(mjcf_path, obj_dir_path)
    
    humanoid_coll = create_collection(motion_name)
    default_collection = bpy.data.collections["Collection"]    

    for j_name in mujoco_joint_names:
        bpy.ops.wm.obj_import(filepath=os.path.join(obj_dir_path, j_name + '.obj'))
        bpy.ops.object.shade_smooth()
        prim = bpy.context.active_object
        prim.name = motion_name + '_' + prim.name
        char_mat = bpy.data.materials.get('char')
        prim.data.materials.append(char_mat)
        move_collection(prim, default_collection, humanoid_coll)


def add_full_body_markers(motion_name='motion'):
    default_collection = bpy.data.collections["Collection"]
    marker_coll = create_collection(motion_name + "_markers")
    for j_name in mujoco_joint_names:
        m = add_marker(motion_name + '_' + 'marker_' + j_name)
        move_collection(m, default_collection, marker_coll)


def add_marker(marker_name):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05)
    bpy.ops.object.shade_smooth()
    obj = bpy.context.active_object
    obj.name = marker_name
    marker_mat = bpy.data.materials.get('marker')
    obj.data.materials.append(marker_mat)
    return obj


def get_object_by_idx(name, idx):
    obj_name = mujoco_joint_names[idx]
    return bpy.data.objects[name + '_' + obj_name]


def smooth_1d(tensor, window_size=3):
    # Ensure window_size is odd
    window_size = max(3, window_size if window_size % 2 == 1 else window_size + 1)
    
    # Create a 1D window for convolution
    window = np.ones(window_size) / window_size
    
    # Pad the first dimension (time axis) of the tensor
    pad_width = [(window_size//2, window_size//2)] + [(0, 0)] * (tensor.ndim - 1)
    padded_tensor = np.pad(tensor, pad_width, mode='edge')
    
    # Apply convolution along the time axis (axis 0)
    smoothed_tensor = np.apply_along_axis(
        lambda x: np.convolve(x, window, mode='valid'), 
        axis=0, 
        arr=padded_tensor
    )
    
    return smoothed_tensor


def smooth_end_effectors(tensor, window_size=3):
    ends_idx = [mujoco_joint_names.index(e) for e in end_effector_names]
    # print(ends_idx)
    # print(tensor.shape, tensor[:, ends_idx, :].shape)
    # exit()
    tensor[:, ends_idx, :] = smooth_1d(tensor[:, ends_idx, :], window_size)
    return tensor


def quaternion_to_euler_angle_vectorized1(quat):
    # quat [bs, wxyz]
    x, y, z, w = np.split(quat, 4, axis=-1)

    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    # X = np.degrees(np.arctan2(t0, t1))
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    # Y = np.degrees(np.arcsin(t2))
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    # Z = np.degrees(np.arctan2(t3, t4))
    Z = np.arctan2(t3, t4)

    return np.concatenate((X, Y, Z), axis=-1)