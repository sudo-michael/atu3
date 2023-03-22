# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import argparse
import collections
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import atu3

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--group", type=str, default="none",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="atu3",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="atu3",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Safe-Air3d-Fixed-Goal-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(1e6),
        help="total timesteps of the experiments")
    parser.add_argument("--save-every", type=int, default=100_000,
        help="save every x steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--policy-noise", type=float, default=0.2,
        help="the scale of policy noise")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=2.5,
        help="TD#j")
    parser.add_argument("--use-bc", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use BC in Loss")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        if idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env = atu3.utils.AutoResetWrapper(env)
    env.seed(seed)
    # env = gym.wrappers.NormalizeObservation(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)
    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=True,
    )
    safe_rb = ReplayBuffer(
        10_000,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(return_info=True)
    obs = np.expand_dims(obs, axis=0) # (1, 3)
    total_unsafe = 0
    total_use_hj = 0
    total_collide_wall = 0
    total_collide_persuer = 0
    total_reach_goal = 0
    success_rate = collections.deque(maxlen=20)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        used_hj = False

        # if global_step < args.learning_starts:
            # actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        if envs.use_opt_ctrl():
            actions = np.expand_dims(envs.opt_ctrl(), 0)
            used_hj = True
            total_use_hj += 1
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.action_space.low, envs.action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, info = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in info.get('terminal_info', {}).keys():
            print(f"global_step={global_step}, episodic_return={info['terminal_info']['episode']['r']} collision={info['terminal_info'].get('collision', 'timeout')}")
            writer.add_scalar("charts/episodic_return", info['terminal_info']["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info['terminal_info']["episode"]["l"], global_step)
            writer.add_scalar("charts/total_hj", total_use_hj, global_step)
            writer.add_scalar("charts/total_unsafe", total_unsafe, global_step)

            if info['terminal_info'].get('TimeLimit.truncated', False):
                success_rate.append(0)
            elif 'collision' in info['terminal_info'].keys():
                if info['terminal_info']['collision'] == 'wall':
                    total_collide_wall += 1
                    success_rate.append(0)
                elif info['terminal_info']['collision'] == 'persuer':
                    total_collide_persuer += 1
                    success_rate.append(0)
                elif info['terminal_info']['collision'] == 'goal':
                    total_reach_goal += 1
                    success_rate.append(1)
                elif info['terminal_info']['collision'] == 'none':
                    success_rate.append(0)
                elif info['terminal_info']['collision'] == 'timeout':
                    success_rate.append(0)
                else:
                    success_rate.append(0)

                
                writer.add_scalar("charts/total_collide_wall", total_collide_wall, global_step)
                writer.add_scalar("charts/total_collide_persuer", total_collide_persuer, global_step)
                writer.add_scalar("charts/total_reach_goal", total_reach_goal, global_step)
                writer.add_scalar("charts/total_hj", total_use_hj, global_step)
                writer.add_scalar("charts/total_unsafe", total_unsafe, global_step)
                writer.add_scalar("charts/success_rate", sum(success_rate) / len(success_rate), global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        if dones:
            real_next_obs = info["terminal_observation"]
        real_next_obs = np.expand_dims(real_next_obs, axis=0)
        if used_hj:
            safe_rb.add(obs, real_next_obs, actions, np.array([rewards]), np.array([dones]), [info])
        else:
            rb.add(obs, real_next_obs, actions, np.array([rewards]), np.array([dones]), [info])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.action_space.low[0], envs.action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                # TD3 + BC
                Q = qf1(data.observations, actor(data.observations)).mean()
                if args.use_bc:
                    safe_data = safe_rb.sample(min(args.batch_size, safe_rb.buffer_size)) # incase we don't use HJ alot
                    lmbda = args.alpha / Q.abs().mean().detach()
                    pi = actor(data.observations)
                    actor_loss = - lmbda * Q.mean() + F.mse_loss(pi, safe_data.actions)
                else:
                    actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
