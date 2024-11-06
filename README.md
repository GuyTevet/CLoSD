# CLoSD: Closing the Loop between Simulation and Diffusion for multi-task character control

[Project Page](https://guytevet.github.io/CLoSD-page/) | [Arxiv](https://arxiv.org/abs/2410.03441) | [Video](https://www.youtube.com/watch?feature=shared&v=O1tzbiDMW8U)

![teaser](https://github.com/GuyTevet/CLoSD-page/blob/main/static/figures/demo1.gif?raw=true)


## Bibtex

If you find this code useful in your research, please cite:

```
@article{tevet2024closd,
  title={CLoSD: Closing the Loop between Simulation and Diffusion for multi-task character control},
  author={Tevet, Guy and Raab, Sigal and Cohan, Setareh and Reda, Daniele and Luo, Zhengyi and Peng, Xue Bin and Bermano, Amit H and van de Panne, Michiel},
  journal={arXiv preprint arXiv:2410.03441},
  year={2024}
}
```


## Getting Started


- The code was tested on `Ubuntu 20.04.5` with `Python 3.8.19`.
- Running CLoSD requires a single GPU with `~4GB RAM` and a monitor.
- Training and evaluation requires a single GPU with `~50GB RAM` (monitor is not required).
- You only need to setup the Python environment. All the dependencies (data, checkpoints, etc.) will be cached automatically on the first run!

<details>
  <summary><b>Setup env</b></summary>

  - Create a Conda env and setup the requirments:

```
conda create -n closd python=3.8
conda activate closd
pip install -r requirement.txt
python -m spacy download en_core_web_sm
```

  - Download [Isaac GYM](https://developer.nvidia.com/isaac-gym), and install it to your env:

```
conda activate closd
cd <ISSAC_GYM_DIR>/python
pip install -e .
```

</details>

## Run CLoSD

<details>
  <summary><b>Multi-task</b></summary>

```
python closd/run.py\
  learning=im_big robot=smpl_humanoid\
  epoch=-1 test=True no_virtual_display=True\
  headless=False env.num_envs=9\
  env=closd_multitask exp_name=CLoSD_multitask_finetune
```

</details>

<details>
  <summary><b>Sequence of tasks</b></summary>
- To be delivered soon
</details>

<details>
  <summary><b>Text-to-motion</b></summary>

```
python closd/run.py\
  learning=im_big robot=smpl_humanoid\
  epoch=-1 test=True no_virtual_display=True\
  headless=False env.num_envs=9\
  env=closd_t2m exp_name=CLoSD_t2m_finetune
```

</details>

- For running with the model witout fine-tuning, use `exp_name=CLoSD_no_finetune`


## Evaluate

<details>
  <summary><b>Multi-task success rate</b></summary>

- To reproduce Table 1 in the paper.

```
python closd/run.py\
 learning=im_big env=closd_multitask robot=smpl_humanoid\
 exp_name=CLoSD_multitask_finetune\
 epoch=-1\
 env.episode_length=500\
 env.dip.cfg_param=7.5\
 env.num_envs=4096\
 test=True\
 no_virtual_display=True\
 headless=True\
 closd_eval=True
```

</details>

<details>
  <summary><b>Text-to-motion</b></summary>

- The raw result are at `https://huggingface.co/guytevet/CLoSD/blob/main/evaluation/closd/eval.log`, this code should reproduce it.
- To reproduce Table 3 in the paper.
- The evaluation process runs on pre-recorded data.

```
python -m closd.diffusion_planner.eval.eval_humanml --external_results_file closd/diffusion_planner/saved_motions/closd/CloSD.pkl --do_unique
```

- To log resutls in Wandb, add:

```
 --train_platform_type WandBPlatform --eval_name <wandb_exp_name>
```

</details>


## Train your own CLoSD

<details>
  <summary><b>Tracking controller (PHC based)</b></summary>

```
python closd/run.py\
 learning=im_big env=im_single_prim robot=smpl_humanoid\
 env.cycle_motion=True epoch=-1\
 exp_name=my_CLoSD_no_finetune
```

- Train for 62K epochs


</details>

<details>
  <summary><b>Fine-tune for Multi-task</b></summary>

```
python closd/run.py\
 learning=im_big env=closd_multitask robot=smpl_humanoid\
 learning.params.load_checkpoint=True\
 learning.params.load_path=output/CLoSD/my_CLoSD_no_finetune/Humanoid.pth\
 env.dip.cfg_param=2.5 env.num_envs=3072\
 has_eval=False epoch=-1\
 exp_name=my_CLoSD_multitask_finetune
```

- Train for 4K epochs

</details>


<details>
  <summary><b>Fine-tune for Text-to-motion</b></summary>

```
python closd/run.py\
 learning=im_big env=closd_t2m robot=smpl_humanoid\
 learning.params.load_checkpoint=True\
 learning.params.load_path=output/CLoSD/my_CLoSD_no_finetune/Humanoid.pth\
 env.dip.cfg_param=2.5 env.num_envs=3072\
 has_eval=False epoch=-1\
 exp_name=my_CLoSD_t2m_finetune
```

- Train for 1K epochs

</details>

- For debug run, use `learning=im_toy` and add `no_log=True env.num_envs=4`

## DiP

- Diffusion Planner (DiP) is a real-time autoregressive diffusion model that serves as the planner for the CLoSD agent.
- Instead of running it as part of CLoSD, you can also run DiP in a stand-alone mode, fed by its own generated motions.
- The following fetails how to sample/evaluate/train DiP in the **stand-alone** mode.

### 

<details>
  <summary><b>Generate Motion with the Stand-alone DiP</b></summary>

Full autoregressive generation (without target):

```
python -m closd.diffusion_planner.sample.generate\
 --model_path closd/diffusion_planner/save/DiP_no-target_10steps_context20_predict40/model000200000.pt\
 --num_repetitions 1 --autoregressive
```

Prefix completion with target trajectory:

```
python -m closd.diffusion_planner.sample.generate\
 --model_path closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt\
 --num_repetitions 1 --sampling_mode goal\
 --target_joint_names "traj,heading" --target_joint_source data
```

- To sample with random joint target (instead of sampling it from the data, which is more challenging), use `--target_joint_source random`
- Other 'legal' joint conditions are:

```
--target_joint_names 
[traj,heading|
pelvis,heading|
right_wrist,heading|
left_wrist,heading|
right_foot,heading|
left_foot,heading]
```

</details>

<details>
  <summary><b>Stand-alone Evaluation</b></summary>

- Evaluate DiP fed by its on predictions (without the CLoSD framework):
- To reproduce Table 2 and 3 (the DiP entry) in the paper.

```
python -m closd.diffusion_planner.eval.eval_humanml\
 --guidance_param 7.5\
 --model_path closd/diffusion_planner/save/DiP_no-target_10steps_context20_predict40/model000600343.pt\
 --autoregressive
```

</details>


<details>
  <summary><b>Train your own DiP</b></summary>

The following will reproduce the DiP used in the paper:

```
python -m closd.diffusion_planner.train.train_mdm\
 --save_dir closd/diffusion_planner/save/my_DiP\
 --dataset humanml --arch trans_dec --text_encoder_type bert\
 --diffusion_steps 10 --context_len 20 --pred_len 40\
 --mask_frames --eval_during_training --gen_during_training --overwrite --use_ema --autoregressive --train_platform_type WandBPlatform
```

To train DiP without target conditioning, add `--lambda_target_loc 0`

</details>

## Acknowledgments

This code is standing on the shoulders of giants. We want to thank the following contributors
that our code is based on:

[MDM](https://github.com/GuyTevet/motion-diffusion-model), [PHC](https://github.com/ZhengyiLuo/PHC), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [actor](https://github.com/Mathux/ACTOR), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [MoDi](https://github.com/sigal-raab/MoDi).

## License
This code is distributed under an [MIT LICENSE](LICENSE).