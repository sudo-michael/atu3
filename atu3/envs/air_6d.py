import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from atu3.utils import normalize_angle, spa_deriv
from atu3.brt.brt_air_3d import grid as backup_grid
from atu3.brt.brt_air_6d import car_brt, grid
import torch

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../siren-reach'))

# siren-reach
#import modules
#from modules import Sine


class Air6dEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, fixed_goal=False, walls=False, penalize_jerk=False, version=1) -> None:
        self.fixed_goal = fixed_goal
        self.walls=walls
        self.car = car_brt
        self.penalize_jerk = penalize_jerk
        self.dt = 0.05

        self.action_space = gym.spaces.Box(
            low=-self.car.we_max, high=self.car.we_max, dtype=np.float32, shape=(1,)
        )

        if self.walls:
            self.observation_space = gym.spaces.Box(
                low=np.array([-4.5, -4.5, -1, -1, -4.5, -4.5]),
                high=np.array([4.5, 4.5, 1, 1, 4.5, 4.5]),
                dtype=np.float32
            )
            # world
            self.world_width = 10
            self.world_height = 10
            self.left_wall = -4.5
            self.right_wall = 4.5
            self.bottom_wall = -4.5
            self.top_wall = 4.5
        else:
            self.observation_space = gym.spaces.Box(
                low=np.array([-10, -10, -1, -1, -10, -10]),
                high=np.array([10, 10, 1, 1, 10, 10]),
                # low=np.array([-5, -5, -1, -1, -5, -5]),
                # high=np.array([5, 5, 1, 1, 5, 5]),
                dtype=np.float32
            )
            self.world_width = 10
            self.world_height = 10
            self.left_wall = -10.0
            self.right_wall = 10.0
            self.bottom_wall = -10.0
            self.top_wall = 10.0
        
        # state
        self.evader_state = np.array([1, 1, 0])
        self.persuer_state = np.array([-1, -1, 0])
        self.goal_location = np.array([2, 2, 0.2])

        self.world_boundary = np.array([2, 2, np.pi], dtype=np.float32)

        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        self.brt = np.load(os.path.join(dir_path, f"assets/brts/air6d_brt_no_wall_5_40_v{version}.npy"))
        self.backup_brt = np.load(os.path.join(dir_path, f"assets/brts/backup_air3d_brt_{version}.npy"))

        self.grid = grid
        self.backup_grid = backup_grid

    def step(self, action):
        # TODO change to odeint
        self.evader_state = (
            self.car.single_dynamics_non_hcl(0, self.evader_state, action, is_evader=True)
            * self.dt
            + self.evader_state
        )
        self.evader_state[2] = normalize_angle(self.evader_state[2])
        self.persuer_state = (
            self.car.single_dynamics_non_hcl(0, self.persuer_state, self.opt_dstb(), is_evader=False)
            * self.dt
            + self.persuer_state
        )
        self.persuer_state[2] = normalize_angle(self.persuer_state[2])

        dist_to_goal = np.linalg.norm(self.evader_state[:2] - self.goal_location[:2])
        reward = (self.last_dist_to_goal - dist_to_goal) * 1.1
        self.last_dist_to_goal = dist_to_goal
        done = False
        info = {}
        info["cost"] = 0
        info["safe"] = True

        if not self.in_bounds(self.evader_state):
            done = True
            if self.walls:
                info["safe"] = False
                info["collision"] = "wall"
                info["cost"] = 1
                reward = -250
            else:
                info['collision'] = 'timeout'
        elif self.near_persuer(self.evader_state, self.persuer_state):
            # breakpoint()
            print('collision')
            print(f"{self.evader_state=}")
            print(f"{self.persuer_state=}")
            print(self.relative_state(self.persuer_state, self.evader_state))
            done = True
            info["safe"] = False
            info["collision"] = "persuer"
            info["cost"] = 1
            reward = -1
        elif self.near_goal(self.evader_state, self.goal_location):
            done = True
            info["collision"] = 'goal'
            # info['steps_to_goal'] = self.t - self.last_t_at_goal
            # self.last_t_at_goal = self.t
            reward = 1
            # self.generate_new_goal_location(self.evader_state)

        info["obs"] = np.copy(self.theta_to_cos_sin(self.evader_state))
        info["persuer"] = np.copy(self.theta_to_cos_sin(self.persuer_state))
        info["goal"] = np.copy(self.goal_location[:2])
        info["brt_value"] = self.grid.get_value(self.brt, self.state(self.persuer_state, self.evader_state))

        return np.copy(self.get_obs(info['obs'], info['persuer'], info['goal'])), reward, done, info

    def reset(self, seed=None, return_info=True):
        if self.fixed_goal:
            self.goal_location = np.array([0.5, 0.5])
        else:
            goal_bounds = np.array([2.5, 2.5])
            self.goal_location = np.random.uniform(
                low=-goal_bounds, high=goal_bounds
            )

        while True:
            self.persuer_state = np.random.uniform(
                low=-self.world_boundary, high=self.world_boundary
            )

            if not self.near_goal(self.persuer_state, self.goal_location, 1.0):
                break
            

        if self.fixed_goal:
            i = 0
            while True and i < 100:
                i += 1
                self.evader_state = np.random.uniform(
                    low=np.array([-2, -2, -np.pi]), high=np.array([-1.0, -1.0, np.pi])
                )

                if self.grid.get_value(
                    self.brt, self.relative_state(self.persuer_state, self.evader_state) 
                ) > 0.2 and not self.near_goal(self.evader_state, self.goal_location):
                    break
        else:
            while True:
                self.evader_state = np.random.uniform(
                    low=-self.world_boundary, high=self.world_boundary
                )

                if self.grid.get_value(
                    self.brt, self.state(self.persuer_state, self.evader_state)
                ) > 0.3 and not self.near_goal(self.evader_state, self.goal_location):
                    break

        # self.evader_state = np.array([0, 0, 0])
        # self.persuer_state = np.array([2, 0, 0])
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
        # self.evader_state = np.array([0, 0, np.pi/2])
        # self.persuer_state = np.array([-1, -3, -np.pi/4])
        # self.evader_state = np.array([0, 0, 0])
        # self.persuer_state = np.array([-1, -1, -3 * np.pi/4])
        # self.persuer_state = np.array([-1, 0, np.pi])
        # self.evader_state = np.array([0, 0, np.pi/4])
        # self.persuer_state = np.array([-1.2, 0, -np.pi/4])
        # self.persuer_state = np.array([-1.2, -1.2, -np.pi/2])
        # self.persuer_state = np.array([-1.2, 1.2, -np.pi/2])
        self.last_dist_to_goal = np.linalg.norm(self.evader_state[:2] - self.goal_location[:2])
        info = {
            "obs": np.copy(self.evader_state),
            "persuer": np.copy(self.persuer_state),
            "goal": np.copy(self.goal_location[:2]),
            "brt_value": self.grid.get_value(self.brt, self.evader_state)
        }
        
        # self.evader_state = np.array([0, 0, -np.pi * 3/4])
        # self.persuer_state = np.array([-2, -2, 0])
        return np.copy(self.get_obs(info['obs'], info['persuer'], info['goal'])), info

    def generate_new_goal_location(self, evader_state):
        while True:
            self.goal_location = np.random.uniform(
                low=-self.world_boundary, high=self.world_boundary
            )

            if self.grid.get_value(
                self.brt, self.goal_location
            ) > 0.3 and not self.near_goal(evader_state, self.goal_location):
                break

    def in_bounds(self, evader_state):
        # if not (
        #     self.left_wall + self.car.r
        #     <= evader_state[0]
        #     <= self.right_wall - self.car.r
        # ):
        #     return False
        # elif not (
        #     self.bottom_wall + self.car.r
        #     <= evader_state[1]
        #     <= self.top_wall - self.car.r
        # ):
        #     return False
        return True

    def near_goal(self, evader_state, goal_state, tol=None):
        # r of goal == self.car.r
        if tol == None:
            return (
                np.linalg.norm(evader_state[:2] - goal_state[:2]) <= self.car.r + self.car.r
            )
        else:
            return (
                np.linalg.norm(evader_state[:2] - goal_state[:2]) <= tol
            )

    def near_persuer(self, evader_state, persuer_state):
        return (
            np.linalg.norm(evader_state[:2] - persuer_state[:2])
            <= self.car.r + self.car.r
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
        add_robot(self.persuer_state, color="red")
        goal = plt.Circle(
            self.goal_location[:2], radius=self.car.r, color="g"
        )
        self.ax.add_patch(goal)

        # walls
        if self.walls:
            self.ax.hlines(y=[-4.5, 4.5], xmin=[-4.5, -4.5], xmax=[4.5, 4.5], color="k")
            self.ax.vlines(x=[-4.5, 4.5], ymin=[-4.5, -4.5], ymax=[4.5, 4.5], color="k")

        X, Y = np.meshgrid(
            np.linspace(self.grid.min[3], self.grid.max[3], self.grid.pts_each_dim[3]), # evader
            np.linspace(self.grid.min[3], self.grid.max[3], self.grid.pts_each_dim[3]),
            indexing="ij",
        )

        state = self.state(self.persuer_state, self.evader_state)
        index = self.grid.get_index(state)
        # angle = (self.evader_state[2] % (2 * np.pi))
        # Xr = X * np.cos(angle) - Y * np.sin(angle)
        # Yr = X * np.sin(angle) + Y * np.cos(angle)

        # self.ax.contour(
        #     Xr + self.evader_state[0],
        #     Yr + self.evader_state[1],
        #     self.brt[:, :, index[5]],
        #     levels=[0.1],
        # )
        self.ax.contour(
            X,
            Y,
            self.brt[index[0], index[1], index[2], :, :, index[5]],
            levels=[0.1],
        )

        if self.walls:
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
        else:
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

    def state(self, persuer_state, evader_state):
        return np.hstack((persuer_state, evader_state))

    def relative_state2(self, persuer_state, evader_state):
        # backup relative state for backup brt when dV/dtheta == 0
        relative_state = persuer_state - evader_state
        relative_state[2] = persuer_state[2]
        return relative_state

    def use_opt_ctrl(self, threshold=0.2):
        # print(f'v: {self.grid.get_value(self.brt, self.state(self.persuer_state, self.evader_state))}')
        return self.grid.get_value(self.brt, self.state(self.persuer_state, self.evader_state)) < threshold

    def opt_ctrl(self):
        # assert -np.pi <= self.evader_state[2] <= np.pi
        state = self.state(self.persuer_state, self.evader_state)
        index = self.grid.get_index(state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        opt_ctrl = self.car.opt_ctrl_non_hcl(state, spat_deriv)
        return opt_ctrl
    
    def opt_dstb(self):
        state = self.state(self.persuer_state, self.evader_state)
        index = self.grid.get_index(state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        if spat_deriv[2] == 0:
            relative_state = self.relative_state2(self.persuer_state, self.evader_state)
            index = self.backup_grid.get_index(relative_state)
            spat_deriv = spa_deriv(index, self.backup_brt, self.backup_grid)
        # print('b: ', spat_deriv[:3])
        opt_dstb = self.car.opt_dstb_non_hcl(spat_deriv)
        return opt_dstb
        

    def relative_state(self, persuer_state, evader_state):
        return np.concatenate((persuer_state, evader_state))
    def get_obs(self, evader_state, persuer_state, goal):
        relative_state = self.relative_state(persuer_state, evader_state)
        relative_state = self.theta_to_cos_sin(relative_state)
        relative_goal = evader_state[:2] - goal[:2]
        return np.concatenate((relative_state, relative_goal))
        
    def theta_to_cos_sin(self, state):
        return np.array(
            [state[0], state[1], np.cos(state[2]), np.sin(state[2])], dtype=np.float32
        )


if __name__ in "__main__":
    import atu3
    from datetime import datetime

    run_name = (
        f"debug__{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    gym.logger.set_level(10)

    # env = gym.make("Safe-Air3d-NoWalls-v0")
    # env = gym.wrappers.TimeLimit(env, 100)
    # env = gym.wrappers.RecordVideo(env, f"debug_videos/{run_name}", episode_trigger=lambda x: True)
    # env = gym.make("Safe-Air3d-v0")
    env = Air6dEnv(walls=False, fixed_goal=True)
    obs = env.reset()
    done = False
    while not done:
        if env.use_opt_ctrl():
            print("using opt ctrl")
            action = env.opt_ctrl()
        
        else:
            action = np.array([-1.0])
        _, _, _, info = env.step(action)

        obs, reward, done, info = env.step(action)
        done = False
        if done:
            print(info)
            print('done')
            break

        env.render()
    env.close()
