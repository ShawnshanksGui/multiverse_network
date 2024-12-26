import torch
import madrona_escape_room
import time
import numpy as np

from madrona_escape_room_learn import (
    train, profile, TrainConfig, PPOConfig, SimInterface,
)

from policy import make_policy, setup_obs

import argparse
import math
from pathlib import Path
import warnings
# warnings.filterwarnings("error")

torch.manual_seed(0)

class LearningCallback:
    def __init__(self, ckpt_dir, profile_report):
        self.mean_fps = 0
        self.ckpt_dir = ckpt_dir
        self.profile_report = profile_report

    def __call__(self, update_idx, update_time, update_results, learning_state):
        update_id = update_idx + 1
        fps = args.num_worlds * args.steps_per_update / update_time
        self.mean_fps += (fps - self.mean_fps) / update_id

        if update_id != 1 and  update_id % 10 != 0:
            return

        ppo = update_results.ppo_stats

        with torch.no_grad():
            reward_mean = update_results.rewards.mean().cpu().item()
            reward_min = update_results.rewards.min().cpu().item()
            reward_max = update_results.rewards.max().cpu().item()

            value_mean = update_results.values.mean().cpu().item()
            value_min = update_results.values.min().cpu().item()
            value_max = update_results.values.max().cpu().item()

            advantage_mean = update_results.advantages.mean().cpu().item()
            advantage_min = update_results.advantages.min().cpu().item()
            advantage_max = update_results.advantages.max().cpu().item()

            bootstrap_value_mean = update_results.bootstrap_values.mean().cpu().item()
            bootstrap_value_min = update_results.bootstrap_values.min().cpu().item()
            bootstrap_value_max = update_results.bootstrap_values.max().cpu().item()

            vnorm_mu = learning_state.value_normalizer.mu.cpu().item()
            vnorm_sigma = learning_state.value_normalizer.sigma.cpu().item()

        print(f"\nUpdate: {update_id}")
        print(f"    Loss: {ppo.loss: .3e}, A: {ppo.action_loss: .3e}, V: {ppo.value_loss: .3e}, E: {ppo.entropy_loss: .3e}")
        print()
        print(f"    Rewards          => Avg: {reward_mean: .3e}, Min: {reward_min: .3e}, Max: {reward_max: .3e}")
        print(f"    Values           => Avg: {value_mean: .3e}, Min: {value_min: .3e}, Max: {value_max: .3e}")
        print(f"    Advantages       => Avg: {advantage_mean: .3e}, Min: {advantage_min: .3e}, Max: {advantage_max: .3e}")
        print(f"    Bootstrap Values => Avg: {bootstrap_value_mean: .3e}, Min: {bootstrap_value_min: .3e}, Max: {bootstrap_value_max: .3e}")
        print(f"    Returns          => Avg: {ppo.returns_mean}, σ: {ppo.returns_stddev}")
        print(f"    Value Normalizer => Mean: {vnorm_mu: .3e}, σ: {vnorm_sigma :.3e}")

        if self.profile_report:
            print()
            print(f"    FPS: {fps:.0f}, Update Time: {update_time:.2f}, Avg FPS: {self.mean_fps:.0f}")
            print(f"    PyTorch Memory Usage: {torch.cuda.memory_reserved() / 1024 / 1024 / 1024:.3f}GB (Reserved), {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.3f}GB (Current)")
            profile.report()

        if update_id % 100 == 0:
            learning_state.save(update_idx, self.ckpt_dir / f"{update_id}.pth")


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=1)
arg_parser.add_argument('--ckpt-dir', type=str, required=True)
arg_parser.add_argument('--restore', type=int)

arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--steps-per-update', type=int, default=40)
arg_parser.add_argument('--num-bptt-chunks', type=int, default=8)

arg_parser.add_argument('--lr', type=float, default=1e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.01)
arg_parser.add_argument('--value-loss-coef', type=float, default=0.5)
arg_parser.add_argument('--clip-value-loss', action='store_true')

arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--profile-report', action='store_true')


# arg_parser.add_argument('--enable_gpu_sim', type=str, required=True, help='Enable GPU simulation (e.g., "cpu" or "gpu")')
arg_parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID')
arg_parser.add_argument('--fattree_K', type=int, required=True, help='Fattree K value')
arg_parser.add_argument('--cc_method', type=int, required=True, help='Congestion control method')

args = arg_parser.parse_args()

# print
# print(f"Number of environments: {args.num_env}")
# print(f"Enable GPU simulation: {args.enable_gpu_sim}")
print(f"gpu id id: {args.gpu_id}")
print(f"Fattree K: {args.fattree_K}")
print(f"CC Method: {args.cc_method}")


class AJLink:
    def __init__(self):
        # 使用NumPy数组来定义一个二维数组，大小为 (100000, 5)
        self.aj_link = np.zeros((100000, 5), dtype=np.uint32)
        self.link_num = 0
        self.npu_num = 0


# 示例用法
aj_link_instance = AJLink()
aj_link_instance.link_num = 10
aj_link_instance.npu_num = 2

# 设置某个链接的值
aj_link_instance.aj_link[0] = [1, 2, 1, 2, 3]


sim = madrona_escape_room.SimManager(
    exec_mode = madrona_escape_room.madrona.ExecMode.CUDA if args.gpu_sim else madrona_escape_room.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    rand_seed = 5,
    auto_reset = True,
    #
    k_aray = args.fattree_K,
    cc_method = args.cc_method,
    # aj_link = aj_link_instance,
    #
)

ckpt_dir = Path(args.ckpt_dir)

learning_cb = LearningCallback(ckpt_dir, args.profile_report)

if torch.cuda.is_available():
    dev = torch.device(f'cuda:{args.gpu_id}')
else:
    dev = torch.device('cpu')

ckpt_dir.mkdir(exist_ok=True, parents=True)

obs, num_obs_features = setup_obs(sim)
policy = make_policy(num_obs_features, args.num_channels, args.separate_value)

actions = sim.action_tensor().to_torch()
dones = sim.done_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()

# Flatten N, A, ... tensors to N * A, ...
actions = actions.view(-1, *actions.shape[2:])
dones  = dones.view(-1, *dones.shape[2:])
rewards = rewards.view(-1, *rewards.shape[2:])

if args.restore:
    restore_ckpt = ckpt_dir / f"{args.restore}.pth"
else:
    restore_ckpt = None


start = time.time()
for i in range(args.num_updates):
    print("----", i, "-th frame----")
    sim.step()

end = time.time()

print("time: ", (end - start))

# train(
#     dev,
#     SimInterface(
#             step = lambda: sim.step(),
#             obs = obs,
#             actions = actions,
#             dones = dones,
#             rewards = rewards,
#     ),
#     TrainConfig(
#         num_updates = args.num_updates,
#         steps_per_update = args.steps_per_update,
#         num_bptt_chunks = args.num_bptt_chunks,
#         lr = args.lr,
#         gamma = args.gamma,
#         gae_lambda = 0.95,
#         ppo = PPOConfig(
#             num_mini_batches=1,
#             clip_coef=0.2,
#             value_loss_coef=args.value_loss_coef,
#             entropy_coef=args.entropy_loss_coef,
#             max_grad_norm=0.5,
#             num_epochs=2,
#             clip_value_loss=args.clip_value_loss,
#         ),
#         value_normalizer_decay = 0.999,
#         mixed_precision = args.fp16,
#     ),
#     policy,
#     learning_cb,
#     restore_ckpt
# )
