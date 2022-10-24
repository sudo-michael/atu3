import gym
import numpy as np
import matplotlib.pyplot as plt
from atu3.utils import normalize_angle, spa_deriv
import os
from atu3.brt.brt_static_obstacle_3d import g as grid
from atu3.brt.brt_static_obstacle_3d import car_brt
from atu3.brt.brt_static_obstacle_3d import cylinder_r
from scipy.integrate import odeint


class StaticObstacle3dEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self) -> None:
        self.car = car_brt
        self.dt = 0.05

        self.action_space = gym.spaces.Box(
            low=-self.car.wMax, high=self.car.wMax, dtype=np.float32, shape=(1,)
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-4.5, -4.5, -1, -1, -4.5, -4.5]),
            high=np.array([4.5, 4.5, 1, 1, 4.5, 4.5]),
            dtype=np.float32
        )
        # state
        self.persuer_state = np.array([0, 0, 0])
        self.evader_state = np.array([1, 1, -1])
        self.goal_location = np.array([2, 2, 0.5])
        # world
        self.world_width = 10
        self.world_height = 10
        self.left_wall = -4.5
        self.right_wall = 4.5
        self.bottom_wall = -4.5
        self.top_wall = 4.5

        self.world_boundary = np.array([4.5, 4.5, np.pi], dtype=np.float32)

        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        self.brt = np.load(
            os.path.join(dir_path, "assets/brts/static_obstacle_brt.npy")
        )
        self.grid = grid
        self.t = 0
        self.last_t_at_goal = 0

    def step(self, action):
        self.t += 1
        sol = odeint(
            self.car.dynamics_non_hcl,
            y0=self.evader_state,
            t=np.linspace(0, self.dt, 4),
            args=(action, np.zeros(3)),
            tfirst=True,
        )
        self.evader_state = sol[-1]
        self.evader_state[2] = normalize_angle(self.evader_state[2])

        reward = -np.linalg.norm(self.evader_state[:2] - self.goal_location[:2])
        done = False
        info = {}
        info["brt_value"] = self.grid.get_value(self.brt, self.evader_state)
        info["cost"] = 0
        info["safe"] = True

        if not self.in_bounds(self.evader_state):
            done = True
            info["safe"] = False
            info["collision"] = "wall"
            info["cost"] = 1
        elif self.near_persuer(self.evader_state, self.persuer_state):
            done = True
            info["safe"] = False
            info["collision"] = "persuer"
            info["cost"] = 1
        elif self.near_goal(self.evader_state, self.goal_location):
            done = True
            info["collision"] = 'goal'
            info['steps_to_goal'] = self.t - self.last_t_at_goal
            self.last_t_at_goal = self.t
            reward = 100
            # self.generate_new_goal_location(self.evader_state)

        info['obs'] = np.copy(self.theta_to_cos_sin(self.evader_state))
        info['goal'] = np.copy(self.goal_location[:2])

        return np.copy(np.concatenate((info['obs'], info['goal']))), reward, done, info

    def reset(self, seed=None):
        while True:
            self.goal_location = np.random.uniform(
                low=-self.world_boundary, high=self.world_boundary
            )

            if self.grid.get_value(self.brt, self.goal_location) > 0.3:
                break

        while True:
            self.evader_state = np.random.uniform(
                low=-self.world_boundary, high=self.world_boundary
            )

            if self.grid.get_value(
                self.brt, self.evader_state
            ) > 0.3 and not self.near_goal(self.evader_state, self.goal_location):
                break

        info = {
            "obs": np.copy(self.theta_to_cos_sin(self.evader_state)),
            "goal": np.copy(self.goal_location[:2]),
            "brt_value": self.grid.get_value(self.brt, self.evader_state)
        }
        return np.copy(np.concatenate((info['obs'], info['goal']))), info

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

    def near_goal(self, evader_state, goal_state):
        # r of goal == self.car.r
        return (
            np.linalg.norm(evader_state[:2] - goal_state[:2]) <= self.car.r + self.car.r
        )

    def near_persuer(self, evader_state, persuer_state):
        return (
            np.linalg.norm(evader_state[:2] - persuer_state[:2])
            <= self.car.r + self.car.r
        )

    def render(self, mode="human"):
        self.ax.clear()

        def add_robot(state, r, color="green"):
            self.ax.add_patch(plt.Circle(state[:2], radius=r, color=color))

            dir = state[:2] + r * np.array([np.cos(state[2]), np.sin(state[2])])

            self.ax.plot([state[0], dir[0]], [state[1], dir[1]], color="c")

        add_robot(self.evader_state, color="blue", r=self.car.r)
        add_robot(self.persuer_state, color="red", r=cylinder_r)
        goal = plt.Circle(self.goal_location[:2], radius=self.car.r, color="g")
        self.ax.add_patch(goal)

        # walls
        self.ax.hlines(y=[-4.5, 4.5], xmin=[-4.5, -4.5], xmax=[4.5, 4.5], color="k")
        self.ax.vlines(x=[-4.5, 4.5], ymin=[-4.5, -4.5], ymax=[4.5, 4.5], color="k")

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

    def use_opt_ctrl(self, threshold=0.1):
        return self.grid.get_value(self.brt, self.evader_state) < threshold

    def opt_ctrl(self):
        assert -np.pi <= self.evader_state[2] <= np.pi
        index = self.grid.get_index(self.evader_state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        opt_ctrl = self.car.opt_ctrl_non_hcl(0, self.evader_state, spat_deriv)
        return opt_ctrl

    def theta_to_cos_sin(self, state):
        return np.array(
            [state[0], state[1], np.cos(state[2]), np.sin(state[2])], dtype=np.float32
        )


if __name__ in "__main__":
    import atu

    env = gym.make('Safe-StaticObstacle3d-v0')
    obs, info = env.reset()
    env.unwrapped.evader_state = np.array([-3.31275266, -3.58783757,  2.64757049])
    for _ in range(300):
        env.render()
        breakpoint()
        print(f"{info['brt_value']=} {env.unwrapped.evader_state=}")
        if env.use_opt_ctrl():
            print("using opt ctrl")
            action = env.opt_ctrl()
        else:
            action = env.action_space.sample()
        _, _, _, info = env.step(action)
