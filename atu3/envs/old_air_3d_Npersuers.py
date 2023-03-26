import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from atu3.utils import normalize_angle, spa_deriv
from atu3.brt.brt_static_obstacle_3d import g as grid
from atu3.brt.brt_static_obstacle_3d import car_brt, car_brat
from atu3.brt.brt_static_obstacle_3d import obstacle_r, goal_r
import itertools


class StaticAirNdEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, n, use_hj=False) -> None:
        self.car = car_brt
        self.dt = 0.05
        self.use_hj = use_hj
        self.n = n

        self.action_space = gym.spaces.Box(
            low=-self.car.wMax, high=self.car.wMax, dtype=np.float32, shape=(1,)
        )

        self.observation_space = gym.spaces.Box(
            low=np.array([-10, -10, -1, -1] * (self.n + 1) + [-10, -10]),
            high=np.array([10, 10, 1, 1] * (self.n + 1) + [10, 10]),
            dtype=np.float32,
        )

        self.world_width = 10
        self.world_height = 10

        # state
        self.persuer_states = [np.array([-1.5, -1.5, np.pi / 4]) for _ in range(self.n)]
        self.evader_state = np.array([1.0, 1.0, np.pi / 4])
        self.goal_location = np.array([1.5, 1.5])
        self.goal_r = goal_r
        self.obstacle_location = np.array([0.0, 0.0])
        self.obstacle_r = obstacle_r

        self.world_boundary = np.array([4.5, 4.5, np.pi], dtype=np.float32)

        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        self.brt = np.load(
            os.path.join(dir_path, f"assets/brts/static_obstacle_brt.npy")
        )

        self.grid = grid

    def step(self, action):
        breakpoint()
        info = {}
        info["used_hj"] = False
        if self.use_hj and self.use_opt_ctrl():
            action = self.opt_ctrl()
            info["used_hj"] = True

        self.evader_state = (
            self.car.dynamics_non_hcl(0, self.evader_state, action) * self.dt
            + self.evader_state
        )
        self.evader_state[2] = normalize_angle(self.evader_state[2])
        for i in range(self.n):
            persuer_action = self.action_space.sample()
            self.persuer_states[i] = (
                self.car.dynamics_non_hcl(0, self.persuer_states[i], persuer_action) * self.dt
                + self.persuer_states[i]
            )
            self.persuer_states[i][2] = normalize_angle(self.persuer_states[i][2])

        dist_to_goal = np.linalg.norm(self.evader_state[:2] - self.goal_location[:2])
        reward = -dist_to_goal
        done = False

        info["cost"] = 0
        info["collision"] = "none"

        if (
            np.linalg.norm(self.evader_state[:2] - self.goal_location)
            < self.goal_r + self.car.r
        ):
            done = True
            print('reach goal')
            info["collision"] = "goal"
        elif (
            np.linalg.norm(self.evader_state[:2] - self.obstacle_location)
            < self.obstacle_r + self.car.r
        ):
            info["collision"] = "obstacle"
            info["cost"] = 1

        return (
            np.copy(self.get_obs(self.evader_state, self.persuer_states, self.goal_location)),
            reward,
            done,
            info,
        )

    def reset(self, seed=None):
        self.evader_state = np.array([-1.5, -1.5, np.pi / 4])

        for i in range(self.n):
            self.persuer_states[i] = np.random.uniform(
                low=-self.world_boundary, high=self.world_boundary
            )

        goal_locations = [np.array([1.5, 1.5]), np.array([-1.5, 1.5]), np.array([1.5, -1.5])]
        self.goal_location = goal_locations[np.random.randint(0, 3)]
        print(self.goal_location)

        info = {}
        info["cost"] = 0
        info["collision"] = "none"

        return self.get_obs(self.evader_state, self.persuer_states, self.goal_location)

    def render(self, mode="human"):
        self.ax.clear()

        def add_robot(state, color="green"):
            self.ax.add_patch(plt.Circle(state[:2], radius=self.car.r, color=color))

            dir = state[:2] + self.car.r * np.array(
                [np.cos(state[2]), np.sin(state[2])]
            )

            self.ax.plot([state[0], dir[0]], [state[1], dir[1]], color="c")

        add_robot(self.evader_state, color="blue")
        for i in range(self.n):
            add_robot(self.persuer_states[i], color="red")
        goal = plt.Circle(self.goal_location[:2], radius=self.goal_r, color="g")
        self.ax.add_patch(goal)
        obstacle = plt.Circle(self.obstacle_location, radius=self.obstacle_r, color="r")
        self.ax.add_patch(obstacle)

        X, Y = np.meshgrid(
            np.linspace(self.grid.min[0], self.grid.max[0], self.grid.pts_each_dim[0]),
            np.linspace(self.grid.min[1], self.grid.max[1], self.grid.pts_each_dim[1]),
            indexing="ij",
        )

        index = self.grid.get_index(self.evader_state)

        self.ax.contour(
            X,
            Y,
            self.brt[:, :, index[2]],
            levels=[0.1],
        )
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
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

    def use_opt_ctrl(self, threshold=0.2):
        return self.grid.get_value(self.brt, self.evader_state) < threshold

    def opt_ctrl(self):
        # assert -np.pi <= self.evader_state[2] <= np.pi
        index = self.grid.get_index(self.evader_state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        opt_ctrl = self.car.opt_ctrl_non_hcl(0, self.evader_state, spat_deriv)
        return opt_ctrl

    def get_obs(self, evader_state, persuer_states, goal):
        t = ([self.theta_to_cos_sin(evader_state)], list(map(self.theta_to_cos_sin, persuer_states)), [goal[:2]])
        return np.concatenate(
            (
                tuple(itertools.chain.from_iterable(t))
            )
        )

    def theta_to_cos_sin(self, state):
        return np.array(
            [state[0], state[1], np.cos(state[2]), np.sin(state[2])], dtype=np.float32
        )


if __name__ in "__main__":
    import atu3
    from datetime import datetime

    run_name = f"debug__{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"

    gym.logger.set_level(10)

    env = StaticAir3dEnv()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
    obs = env.reset()
    print(obs.shape)
    exit()
    done = False
    while not done:
        if env.use_opt_ctrl():
            print("using opt ctrl")
            action = env.opt_ctrl()
        else:
            action = env.action_space.sample()
        _, _, _, info = env.step(action)

        obs, reward, done, info = env.step(action)
        print(reward)
        if done:
            print(info)
            print("done")
            break

        env.render()
    env.close()
