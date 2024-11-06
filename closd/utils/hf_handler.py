from huggingface_hub import hf_hub_download
from huggingface_hub._snapshot_download import snapshot_download
import os

REPO_NAME = 'guytevet/CLoSD'

def get_dependencies():
    dependencies_path = snapshot_download(repo_id=REPO_NAME)
    print('Data dependencies are cached at [{}]'.format(dependencies_path))
    link_all_checkpoints(dependencies_path)
    return dependencies_path


def link_all_checkpoints(dependencies_path):
    link_checkpoints(os.path.join(dependencies_path, 'checkpoints', 'dip'), os.path.join('closd','diffusion_planner','save'))  # DiP checkpoints
    link_checkpoints(os.path.join(dependencies_path, 'evaluation'), os.path.join('closd','diffusion_planner','saved_motions'))  # DiP checkpoints
    link_checkpoints(os.path.join(dependencies_path, 'checkpoints', 'closd'), os.path.join('output','CLoSD'))  # CLoSD checkpoints


def link_checkpoints(src_dir, dst_dir):
    assert os.path.isdir(src_dir)
    os.makedirs(dst_dir, exist_ok=True)
    all_subdirs = [subdir for subdir in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, subdir))]
    for subdir in all_subdirs:
        if not os.path.exists(os.path.join(dst_dir, subdir)):
            os.symlink(os.path.join(src_dir, subdir), os.path.join(dst_dir, subdir))