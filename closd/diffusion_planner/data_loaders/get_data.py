from torch.utils.data import DataLoader
from closd.diffusion_planner.data_loaders.tensors import collate as all_collate
from closd.diffusion_planner.data_loaders.tensors import t2m_collate, t2m_prefix_collate
from closd.utils import hf_handler

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from closd.diffusion_planner.data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from closd.diffusion_planner.data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train', pred_len=0, batch_size=1):
    if hml_mode == 'gt':
        from closd.diffusion_planner.data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        if pred_len > 0:
            return lambda x: t2m_prefix_collate(x, pred_len=pred_len)
        return lambda x: t2m_collate(x, batch_size)
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', abs_path='.', fixed_len=0, hml_type=None, 
                device=None, autoregressive=False, return_keys=False, cache_path=None): 
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, abs_path=abs_path, fixed_len=fixed_len, hml_type=hml_type, 
                       device=device, autoregressive=autoregressive, return_keys=return_keys, cache_path=cache_path)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', hml_type=None, abs_path='closd/diffusion_planner', fixed_len=0, pred_len=0, 
                       device=None, autoregressive=False, drop_last=True, return_keys=False):
    cache_path = hf_handler.get_dependencies()
    dataset = get_dataset(name, num_frames, split, hml_mode, abs_path, fixed_len, hml_type, device, autoregressive, return_keys, cache_path)
    collate = get_collate_fn(name, hml_mode, pred_len, batch_size)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=drop_last, collate_fn=collate
    )

    return loader

def replace_loader(dataset, old_loader):
    collate = old_loader.collate_fn; batch_size = old_loader.batch_size; drop_last = old_loader.drop_last
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=drop_last, collate_fn=collate
    )
    del old_loader
    return loader
