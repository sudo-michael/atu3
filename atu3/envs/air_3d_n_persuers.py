import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from atu3.utils import normalize_angle, spa_deriv
from atu3.brt.brt_air_3d import car_brt, grid, car_brt_2, car_brt_3
import torch

import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../siren-reach")
)

# load confing into dataclasas like structure thing
import json
with open("./atu3/envs/assets/config/brt_1e2p.json", 'r') as f:
    opt = f.read() 
opt = json.loads(opt)
for k, v in opt.items():
    if k[0] != '_':
        opt[k] = v['value']
opt.pop('_wandb')
from collections import namedtuple
opt = namedtuple('opt', opt.keys())(*opt.values())


import modules, dataio, diff_operators
dataset = dataio.Reachability1E2PSource(numpoints=65000, tMin=opt.tMin, tMax=opt.tMax,
                                        counter_start=opt.counter_start, counter_end=opt.counter_end,
                                        collisionR=opt.collisionR, velocitye=opt.velocitye, velocityp=opt.velocityp, omegaMaxe=opt.omegaMaxe,omegaMaxp=opt.omegaMaxp, diffModel=opt.diffModel,
                                        pretrain=opt.pretrain, pretrain_iters=opt.pretrain_iters, num_src_samples=opt.num_src_samples)
model = modules.SingleBVPNet(in_features=13, out_features=1, type=opt.model, mode=opt.mode,
                            final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl, 
                            input_transform_function=dataset.input_transform_function)
model.cuda()
# torch.load("../../siren-reach/logs/BRTPeriodicityIssue/1E2P/CollisionR_0x2/normalModel_v_with_periodic_transform_t3_v2/checkpoints/model_final.pth")
# model.load_state_dict(torch.load("../../siren-reach/logs/BRTPeriodicityIssue/1E2P/CollisionR_0x2/normalModel_v_with_periodic_transform_t3_v3/checkpoints/model_final.pth"))
model.load_state_dict(torch.load("../../siren-reach/logs/BRTPeriodicityIssue/1E2P/CollisionR_0x2/normalModel_v_with_periodic_transform_t3_v2/checkpoints/model_final.pth"))

class Air3dNpEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, fixed_goal, walls, penalize_jerk=False, version=1, n_persuers=2, use_deepreach=False) -> None:
        self.return_info = True
        self.fixed_goal = fixed_goal
        self.walls = walls
        self.penalize_jerk = penalize_jerk
        self.use_deepreach = use_deepreach
        self.n_persuers = n_persuers

        if version == 1:
            self.car = car_brt
        elif version == 2:
            self.car = car_brt_2
        elif version == 3:
            self.car = car_brt
        self.dt = 0.05

        self.action_space = gym.spaces.Box(
            low=-self.car.we_max, high=self.car.we_max, dtype=np.float32, shape=(1,)
        )

        if self.walls:
            self.observation_space = gym.spaces.Box(
                low=np.array([-4.5, -4.5, -1, -1, -4.5, -4.5, -4.5, -4.5]),
                high=np.array([4.5, 4.5, 1, 1, 4.5, 4.5, -4.5, -4.5]),
                dtype=np.float32,
            )
            # world
            self.world_width = 10
            self.world_height = 10
            self.left_wall = -4.5
            self.right_wall = 4.5
            self.bottom_wall = -4.5
            self.top_wall = 4.5
        else:
            relative_persuers = []
            for _ in range(self.n_persuers):
                relative_persuers.extend([10, 10, 1, 1])

            relative_persuers = np.array(
                relative_persuers
            )  # rel x rel y rel cos theta rel sin theta
            self.observation_space = gym.spaces.Box(
                low=np.concatenate(
                    (-relative_persuers, np.array([-10, -10, -10, -10, -10]))
                ),  # dist to goal x, dist to goal y, dist to goal, goal x goal y
                high=np.concatenate(
                    (relative_persuers, np.array([10, 10, 10, 10, 10]))
                ),  # dist to goal x, dist to goal y, dist to goal, goal x goal y
                dtype=np.float32,
            )
            self.world_width = 10
            self.world_height = 10
            self.left_wall = -10.0
            self.right_wall = 10.0
            self.bottom_wall = -10.0
            self.top_wall = 10.0

        # state
        self.evader_state = np.array([1, 1, 0])
        self.persuer_state = np.zeros((self.n_persuers, 3))
        self.goal_location = np.array([2, 2, 0.2])
        self.goal_r = 0.2

        self.world_boundary = np.array([1.0, 1.0, np.pi], dtype=np.float32)

        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        self.brt = np.load(
            os.path.join(dir_path, f"assets/brts/air3d_brt_{version}.npy")
        )
        self.backup_brt = np.load(
            os.path.join(dir_path, f"assets/brts/backup_air3d_brt_{version}.npy")
        )

        self.grid = grid

    def step(self, action):
        self.evader_state = (
            self.car.dynamics_non_hcl(0, self.evader_state, action, is_evader=True)
            * self.dt
            + self.evader_state
        )
        self.evader_state[2] = normalize_angle(self.evader_state[2])
        # DBG
        obs = self.get_deepreach_obs()
        model_input = {'coords': torch.Tensor(obs).cuda()}
        model_output = model(model_input)

        x = model_output['model_in']  # (meta_batch_size, num_points, 3)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        # print(f'value: {y.item()}')
        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]
        # print(dudx[0, 0, -3:])

        for i, opt_dstb in enumerate(self.opt_dstb()):
            # if i == 1:
            #     continue
            self.persuer_state[i] = (
                self.car.dynamics_non_hcl(
                    0, self.persuer_state[i], opt_dstb, is_evader=False
                )
                * self.dt
                + self.persuer_state[i]
            )
            self.persuer_state[i][2] = normalize_angle(self.persuer_state[i][2])

        dist_to_goal = np.linalg.norm(self.evader_state[:2] - self.goal_location[:2])
        reward = (self.last_dist_to_goal - dist_to_goal) * 1.1
        self.last_dist_to_goal = dist_to_goal

        if self.penalize_jerk:
            reward -= (1/120) * np.abs((action[0] - self.last_theta_dot)) / 0.05 # divide by 60 to make reward not too big
            self.last_theta_dot = action[0]
        done = False
        info = {}
        info["brt_value"] = self.grid.get_value(self.brt, self.evader_state)
        info["cost"] = 0
        info["safe"] = True
        info['collision'] = 'none'

        if not self.in_bounds(self.evader_state):
            done = True
            if self.walls:
                info["safe"] = False
                info["collision"] = "wall"
                info["cost"] = 1
                reward = -250
            else:
                info["collision"] = "timeout"
        elif self.near_persuers(self.evader_state, self.persuer_state):
            print("collision")
            print(f"{self.evader_state=}")
            print(f"{self.persuer_state=}")
            # print(self.relative_state(self.persuer_state, self.evader_state))
            done = True
            info["safe"] = False
            info["collision"] = "persuer"
            info["cost"] = 1
            # reward = -250
        elif self.near_goal(self.evader_state, self.goal_location):
            done = True
            info["collision"] = "goal"
            # info['steps_to_goal'] = self.t - self.last_t_at_goal
            # self.last_t_at_goal = self.t
            reward = 5
            # self.generate_new_goal_location(self.evader_state)

        info["obs"] = np.copy(self.theta_to_cos_sin(self.evader_state))
        info["persuer"] = np.copy(self.theta_to_cos_sin(self.persuer_state, b=True))
        info["goal"] = np.copy(self.goal_location[:2])
        info["brt_value"] = self.grid.get_value(self.brt, self.evader_state)
        return (
            np.copy(self.get_obs(info["obs"], info["persuer"], info["goal"])),
            reward,
            done,
            info,
        )

    def reset(self, seed=None, return_info=True):
        if self.fixed_goal:
            self.goal_location = np.array([0.5, 0.5])
        else:
            goal_bounds = np.array([2.5, 2.5])
            self.goal_location = np.random.uniform(low=-goal_bounds, high=goal_bounds)

        # if self.fixed_goal:
        #     for i in range(self.n_persuers):
        #         while True:
        #             self.persuer_state[i] = np.random.uniform(
        #                 low=np.array([-1, -1, -np.pi]), high=np.array([1, 1, np.pi])
        #             )

        #             if not self.near_goal(self.persuer_state[i], self.goal_location):
        #                 break
        # else:
        #     while True:
        #         self.persuer_state = np.random.uniform(
        #             low=-self.world_boundary, high=self.world_boundary
        #         )

        #         if not self.near_goal(self.persuer_state, self.goal_location, 1.0):
        #             break

        # if self.fixed_goal:
        #     i = 0
        #     while True and i < 10:
        #         i += 1
        #         self.evader_state = np.random.uniform(
        #             low=np.array([-3.5, -3.5, -np.pi]), high=np.array([-3, -3, np.pi])
        #         )

        #         values = np.array([self.grid.get_value(self.brt, self.relative_state(self.persuer_state[i], self.evader_state)) for i in range(self.n_persuers)])
        #         if np.all(
        #                 values > 0.2
        #         ) and not self.near_goal(self.evader_state, self.goal_location):
        #             break
        # else:
        #     while True:
        #         self.evader_state = np.random.uniform(
        #             low=-self.world_boundary, high=self.world_boundary
        #         )

        #         if self.grid.get_value(
        #             self.brt, self.relative_state(self.persuer_state, self.evader_state)
        #         ) > 0.3 and not self.near_goal(self.evader_state, self.goal_location):
        #             break

        self.evader_state = np.array([0, 0, 0])
        self.persuer_state[1] = np.array([3, 0, 0.0])
        self.persuer_state[0] = np.array([-3, 0, -np.pi])
        # self.evader_state = np.array([0, 0, 0])
        # self.persuer_state = np.array([-2, 0, 0])
        # self.evader_state = np.array([0, 0, 0.68])
        # self.persuer_state = np.array([2, 2, -0.30])
        # self.evader_state = np.array([0, 0, -np.pi/2])
        # self.persuer_state = np.array([0, -2, np.pi/2])
        # self.evader_state = np.array([0, 0, np.pi/2])
        # self.persuer_state = np.array([0, 2, -np.pi/2])
        # self.evader_state = np.array([-3, -3, np.pi/4])
        # self.persuer_state = np.array([2, 2, -np.pi])
        # NOT WORKING
        # self.evader_state = np.array([0, 0, -np.pi/2 * 3])
        # self.persuer_state = np.array([-1, -3, -np.pi/4])
        # self.evader_state = np.array([0, 0, 0])
        # self.persuer_state = np.array([-1, -1, -3 * np.pi/4])
        # self.persuer_state = np.array([-1, 0, np.pi])
        # self.evader_state = np.array([0, 0, np.pi/4])
        # self.persuer_state = np.array([-1.2, 0, -np.pi/4])
        # self.persuer_state = np.array([-1.2, -1.2, -np.pi/2])
        # self.persuer_state = np.array([-1.2, 1.2, -np.pi/2])
        self.last_dist_to_goal = np.linalg.norm(
            self.evader_state[:2] - self.goal_location[:2]
        )
        info = {
            "obs": np.copy(self.evader_state),
            "persuer": np.copy(self.persuer_state),
            "goal": np.copy(self.goal_location[:2]),
            "brt_value": self.grid.get_value(self.brt, self.evader_state),
        }
        return np.copy(self.get_obs(info["obs"], info["persuer"], info["goal"])), info

    def generate_new_goal_location(self, evader_state):
        while True:
            self.goal_location = np.random.uniform(
                low=-self.world_boundary, high=self.world_boundary
            )

            if self.grid.get_value(
                self.brt, self.goal_location
            ) > 0.3 and not self.near_goal(evader_state, self.goal_location):
                break

    def get_deepreach_obs(self, persuer_1_first=True):
        alpha = dataset.alpha
        if persuer_1_first:
            x_normalized = np.array([[
                3.0,
                self.evader_state[0], self.evader_state[1],
                self.persuer_state[0][0], self.persuer_state[0][1],
                self.persuer_state[1][0], self.persuer_state[1][1], 
                self.evader_state[2], self.persuer_state[0][2], self.persuer_state[1][2]
            ]])
        else:
            x_normalized = np.array([[
                3.0,
                self.evader_state[0], self.evader_state[1],
                self.persuer_state[1][0], self.persuer_state[1][1],
                self.persuer_state[0][0], self.persuer_state[0][1], 
                self.evader_state[2], self.persuer_state[1][2], self.persuer_state[0][2]
            ]])

        x_normalized[..., 1] = x_normalized[..., 1] / alpha['x']
        x_normalized[..., 2] = x_normalized[..., 2] / alpha['y']
        x_normalized[..., 3] = x_normalized[..., 3] / alpha['x']
        x_normalized[..., 4] = x_normalized[..., 4] / alpha['y']
        x_normalized[..., 5] = x_normalized[..., 5] / alpha['x']
        x_normalized[..., 6] = x_normalized[..., 6] / alpha['y']
        x_normalized[..., 7] = x_normalized[..., 7] / alpha['th']
        x_normalized[..., 8] = x_normalized[..., 8] / alpha['th']
        x_normalized[..., 9] = x_normalized[..., 9] / alpha['th']

        return x_normalized

    def in_bounds(self, evader_state):
        if not (
            self.left_wall + self.car.r
            <= evader_state[0]
            <= self.right_wall - self.car.r
        ):
            return False
        elif not (
            self.bottom_wall + self.car.r
            <= evader_state[1]
            <= self.top_wall - self.car.r
        ):
            return False
        return True

    def near_goal(self, evader_state, goal_state, tol=None):
        # r of goal == self.car.r
        if tol == None:
            return np.linalg.norm(evader_state[:2] - goal_state[:2]) <= (self.goal_r + self.car.r)
        else:
            return np.linalg.norm(evader_state[:2] - goal_state[:2]) <= tol

    def near_persuers(self, evader_state, persuer_state):

        return np.any(
            [
                np.linalg.norm(evader_state[:2] - persuer_state[i][:2])
                <= self.car.r + self.car.r
                for i in range(self.n_persuers)
            ]
        )

    def render(self, mode="human"):
        self.ax.clear()

        def add_robot(state, color="green"):
            self.ax.add_patch(plt.Circle(state[:2], radius=self.car.r, color=color))

            dir = state[:2] + self.car.r * np.array(
                [np.cos(state[2]), np.sin(state[2])]
            )

            self.ax.plot([state[0], dir[0]], [state[1], dir[1]], color="c")

        add_robot(self.evader_state, color="blue")
        for i in range(self.n_persuers):
            if i == 0:
                add_robot(self.persuer_state[i], color="red")
            elif i == 1:
                add_robot(self.persuer_state[i], color="orange")
        goal = plt.Circle(self.goal_location[:2], radius=self.goal_r, color="g")
        # self.ax.add_patch(goal)

        # walls
        if self.walls:
            self.ax.hlines(y=[-4.5, 4.5], xmin=[-4.5, -4.5], xmax=[4.5, 4.5], color="k")
            self.ax.vlines(x=[-4.5, 4.5], ymin=[-4.5, -4.5], ymax=[4.5, 4.5], color="k")

        X, Y = np.meshgrid(
            np.linspace(self.grid.min[0], self.grid.max[0], self.grid.pts_each_dim[0]),
            np.linspace(self.grid.min[1], self.grid.max[1], self.grid.pts_each_dim[1]),
            indexing="ij",
        )

        # obs = self.get_deepreach_obs()[0]
        # t = torch.ones(X.shape[0]**2, 1) * obs[0]
        # # xe = torch.ones(X.shape[0], 1) * obs[1]
        # # ye = torch.ones(X.shape[0], 1) * obs[2]
        # xe = torch.tensor(X.reshape(-1, 1)) / dataset.alpha['x']
        # ye = torch.tensor(Y.reshape(-1, 1)) / dataset.alpha['y']
        # xp = torch.ones(X.shape[0]**2, 1) * obs[3] / dataset.alpha['x']
        # yp = torch.ones(X.shape[0]**2, 1) * obs[4] / dataset.alpha['y']
        # xp2 = torch.ones(X.shape[0]**2, 1) * obs[5] / dataset.alpha['x']
        # yp2 = torch.ones(X.shape[0]**2, 1) * obs[6] / dataset.alpha['y']
        # te = torch.ones(X.shape[0]**2, 1) * obs[7]/ dataset.alpha['th']
        # tp = torch.ones(X.shape[0]**2, 1) * obs[8]/ dataset.alpha['th']
        # tp2 = torch.ones(X.shape[0]**2, 1) * obs[9]/ dataset.alpha['th']
        # # coords = torch.cat((time_coords, mgrid_coords, one_coords / 3.0, zero_coords, -one_coords / 3.0, zero_coords, zero_coords, theta_coords, theta_coords), dim=1) 
        # mgrid_coords = dataio.get_mgrid(sidelen)
        # model_input = {'coords': torch.FloatTensor(coords).cuda()}
        # model_output = model(model_input)
        # breakpoint()

        # relative_state = self.relative_state(self.persuer_state, self.evader_state)
        # index = self.grid.get_index(relative_state)
        # angle = self.evader_state[2] % (2 * np.pi)
        # Xr = X * np.cos(angle) - Y * np.sin(angle)
        # Yr = X * np.sin(angle) + Y * np.cos(angle)

        # self.ax.contour(
        #     Xr + self.evader_state[0],
        #     Yr + self.evader_state[1],
        #     self.brt[:, :, index[2]],
        #     levels=[0.1],
        # )

        # self.ax.contour(
        #     X,
        #     Y,
        #     levels=[0.1],
        # )

        # if self.walls:
        #     self.ax.set_xlim(-5, 5)
        #     self.ax.set_ylim(-5, 5)
        # else:

        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)

        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        if mode == "human":
            self.fig.canvas.flush_events()
            plt.pause(1 / self.metadata["render_fps"])
            # plt.show()
        return img

    def close(self):
        plt.close()
        return

    def relative_state(self, persuer_state, evader_state):
        rotated_relative_state = np.zeros(3)
        relative_state = persuer_state - evader_state

        angle = -evader_state[2]

        # fmt: off
        # brt assume that evader_state is at theta=0
        rotated_relative_state[0] = relative_state[0] * np.cos(angle) - relative_state[1] * np.sin(angle)
        rotated_relative_state[1] = relative_state[0] * np.sin(angle) + relative_state[1] * np.cos(angle)
        # fmt: on

        # after rotating by -evader_state[2], the relative angle will still be the same
        # rotated_relative_state[0] = np.abs(rotated_relative_state[0])
        # rotated_relative_state[1] = np.abs(rotated_relative_state[1])
        rotated_relative_state[2] = normalize_angle(relative_state[2])
        return rotated_relative_state

    def relative_state2(self, persuer_state, evader_state):
        # backup relative state for backup brt when dV/dtheta == 0
        relative_state = persuer_state - evader_state
        relative_state[2] = persuer_state[2]
        return relative_state

    def use_opt_ctrl(self, threshold=0.2):
        value = self.get_value()
        print(value)
        return value < threshold

    def opt_ctrl(self):
        # assert -np.pi <= self.evader_state[2] <= np.pi
        return self.get_opt_ctrl()

    def opt_dstb(self):
        return self.get_opt_dstb()

    def get_obs(self, evader_state, persuer_state, goal):
        relative_states = []
        for i in range(self.n_persuers):
            relative_state = self.relative_state(persuer_state[i], evader_state)
            relative_state = self.theta_to_cos_sin(relative_state)
            relative_states.extend(relative_state)
        relative_states = np.array(relative_states, dtype=np.float32)
        relative_goal = evader_state[:2] - goal[:2]
        dist_to_goal = np.array([np.linalg.norm(relative_goal) - self.goal_r])
        return np.concatenate((relative_states, relative_goal, dist_to_goal, goal[:2]))

    def theta_to_cos_sin(self, state, b=False):
        if b:
            return np.concatenate((state[..., 0], state[..., 1], np.cos(state[..., 2]), np.sin(state[..., 2])))
        return np.array(
            [state[0], state[1], np.cos(state[2]), np.sin(state[2])], dtype=np.float32
        )
    
    def get_opt_dstb(self):
        obs = self.get_deepreach_obs()
        model_input = {'coords': torch.Tensor(obs).cuda()}
        model_output = model(model_input)

        x = model_output['model_in']  # (meta_batch_size, num_points, 3)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]
        _, p1_dstb, _ = dataset.compute_overall_ham(x[..., 1:], dudx, return_opt_ctrl=True)
        p1_dstb = p1_dstb.item()

        obs = self.get_deepreach_obs(persuer_1_first=False)
        model_input = {'coords': torch.Tensor(obs).cuda()}
        model_output = model(model_input)

        x = model_output['model_in']  # (meta_batch_size, num_points, 3)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]
        _, p2_dstb, _ = dataset.compute_overall_ham(x[..., 1:], dudx, return_opt_ctrl=True)
        p2_dstb = p2_dstb.item()
        return np.array([p1_dstb]), np.array([p2_dstb])
    
    def get_opt_ctrl(self):
        obs = self.get_deepreach_obs()
        model_input = {'coords': torch.Tensor(obs).cuda()}
        model_output = model(model_input)

        # opt_disb        
        x = model_output['model_in']  # (meta_batch_size, num_points, 3)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]
        opt_ctrl, _, _ = dataset.compute_vehicle_ham(x[..., 1:], dudx, return_opt_ctrl=True)
        return np.array([opt_ctrl.item()])

    def get_value(self):
        obs = self.get_deepreach_obs()
        # normalize obs
        norm_obs = self.normalize_deepreach_obs(obs)
        with torch.no_grad():
            model_input = {'coords': torch.Tensor(norm_obs).cuda()}
            model_out = model(model_input)
        model_out = model_out['model_out']
        model_out = (model_out*dataset.var/dataset.norm_to) + dataset.mean 
        value = model_out.item()
        return value

    def normalize_deepreach_obs(self, obs):
        obs[..., 1] = obs[..., 1] / dataset.alpha['x']
        obs[..., 2] = obs[..., 2] / dataset.alpha['y']
        obs[..., 3] = obs[..., 3] / dataset.alpha['x']
        obs[..., 4] = obs[..., 4] / dataset.alpha['y']
        obs[..., 5] = obs[..., 5] / dataset.alpha['x']
        obs[..., 6] = obs[..., 6] / dataset.alpha['y']
        obs[..., 7] = obs[..., 7] / dataset.alpha['th']
        obs[..., 8] = obs[..., 8] / dataset.alpha['th']
        obs[..., 9] = obs[..., 9] / dataset.alpha['th']
        return obs
        
        


if __name__ in "__main__":
    import atu3
    from datetime import datetime

    run_name = f"debug__{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"

    gym.logger.set_level(10)

    env = Air3dNpEnv(fixed_goal=True, walls=False, penalize_jerk=False, use_deepreach=True, version=1)
    # env = gym.make("Safe-Air3D-2p-NoWalls-Fixed-v1")
    # env = gym.wrappers.TimeLimit(env, 100)
    env = gym.wrappers.RecordVideo(env, f"debug_videos/{run_name}", episode_trigger=lambda x: True)
    # env = gym.make("Safe-Air3d-v0")
    obs = env.reset()
    # print(obs)
    done = False
    for _ in range(200):
        if env.use_opt_ctrl():
            print("using opt ctrl")
            action = env.opt_ctrl()
        else:
            print("not using opt ctrl")
            action = np.array([1.5])
            # action = env.action_space.sample()
        _, _, _, info = env.step(action)

        obs, reward, done, info = env.step(action)
        if done:
            print(info)
            print("done")
            break

        env.render()
    env.close()
