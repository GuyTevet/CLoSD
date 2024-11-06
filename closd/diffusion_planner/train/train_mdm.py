# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from closd.diffusion_planner.utils.fixseed import fixseed
from closd.diffusion_planner.utils.parser_util import train_args
from closd.diffusion_planner.utils import dist_util
from closd.diffusion_planner.train.training_loop import TrainLoop
from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from closd.diffusion_planner.utils.model_util import create_model_and_diffusion
from closd.diffusion_planner.train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandBPlatform  # required for the eval operation

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")

    data = get_dataset_loader(name=args.dataset, 
                              batch_size=args.batch_size, 
                              num_frames=args.num_frames, 
                              fixed_len=args.pred_len + args.context_len, 
                              pred_len=args.pred_len,
                              hml_type=args.hml_type,
                              device=dist_util.dev(),)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
