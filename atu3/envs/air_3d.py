import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from atu3.utils import normalize_angle, spa_deriv
from atu3.brt.brt_air_3d import car_brt, grid, persuer_backup_brt


class Air3dEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, fixed_goal, penalize_jerk=False, version=0) -> None:
        self.return_info = True
        self.fixed_goal = fixed_goal
        self.penalize_jerk = penalize_jerk
        print(f"{penalize_jerk=}")
        self.car = car_brt
        self.dt = 0.05

        self.action_space = gym.spaces.Box(
            low=-self.car.we_max, high=self.car.we_max, dtype=np.float32, shape=(1,)
        )

        # obs: [x_e, y_e, cos(theta_e) sin(theta_e),
        #       x_p, y_p, cos(theta_p) sin(theta_p),
        #       distance_to_evader_from_persuer
        #       x_g, y_g
        self.observation_space = gym.spaces.Box(
            low=np.array([-10, -10, -1, -1, -10, -10, -1, 1, -10, -10, -10]),
            high=np.array([10, 10, 1, 1, 10, 10, 1, 1, 10, 10, 10]),
            dtype=np.float32,
        )
        
        # check if evader is OOB, terminate episode early if so
        self.left_wall = -2.0
        self.right_wall = 2.0
        self.bottom_wall = -2.0
        self.top_wall = 2.0

        self.world_width = 10
        self.world_height = 10

        self.evader_state = np.array([0, 1, 0])
        self.persuer_state = np.array([-1, 0, 0])

        self.goal_location = np.array([1, 1])
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
        self.persuer_state = (
            self.car.dynamics_non_hcl(0, self.persuer_state, self.opt_dstb(), is_evader=False)
            * self.dt
            + self.persuer_state
        )
        self.persuer_state[2] = normalize_angle(self.persuer_state[2])
        
        dist_to_goal = np.linalg.norm(self.evader_state[:2] - self.goal_location[:2])
        reward = self.last_dist_to_goal - dist_to_goal
        self.last_dist_to_goal = dist_to_goal

        if self.penalize_jerk:
            reward -= (
                (1 / 120) * np.abs((action[0] - self.last_theta_dot)) / 0.05
            )  # divide by 60 to make reward not too big
            self.last_theta_dot = action[0]

        done = False
        info = {}
        info["brt_value"] = self.grid.get_value(self.brt, self.evader_state)
        info["cost"] = 0
        info["safe"] = True
        info["collision"] = "none"

        if not self.in_bounds(self.evader_state):
            done = True
            info["collision"] = "oob"
        elif self.near_persuer(self.evader_state, self.persuer_state):
            print("--- collision ---")
            print(f"evader:  {self.evader_state=}")
            print(f"persuer: {self.persuer_state=}")
            done = True
            info["safe"] = False
            info["collision"] = "persuer"
            info["cost"] = 1
            reward = -1
        elif self.near_goal(self.evader_state, self.goal_location, tol=self.car.r + self.goal_r):
            done = True
            info["collision"] = "goal"
            reward = 1

        info["obs"] = np.copy(self.theta_to_cos_sin(self.evader_state))
        info["persuer"] = np.copy(self.theta_to_cos_sin(self.persuer_state))
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
            self.goal_location = np.array([0, 0])
            while True:
                self.persuer_state = np.random.uniform(
                    low=np.array([-2, -2, -np.pi]), high=np.array([2, 2, np.pi])
                )

                if not self.near_goal(self.persuer_state, self.goal_location, self.car.r + self.goal_r):
                    break

            i = 0
            while True and i < 100:
                i += 1
                self.evader_state = np.random.uniform(
                    low=np.array([-2, -2, -np.pi]), high=np.array([2.0, 2.0, np.pi])
                )

                if self.grid.get_value(
                    self.brt, self.relative_state(self.persuer_state, self.evader_state)
                ) > 0.2 and not self.near_goal(self.evader_state, self.goal_location, self.car.r + self.goal_r):
                    break

        
        # TODO: remove 
        # self.evader_state = np.array([0, 1, 0])
        # self.persuer_state = np.array([-1, 0, 0])

        self.last_dist_to_goal = np.linalg.norm(
            self.evader_state[:2] - self.goal_location[:2]
        )
        self.last_theta_dot = 0
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

    def near_goal(self, evader_state, goal_state, tol):
        return np.linalg.norm(evader_state[:2] - goal_state[:2]) <= tol

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
        goal = plt.Circle(self.goal_location[:2], radius=self.goal_r, color="g")
        self.ax.add_patch(goal)

        self.ax.hlines(y=[-2, 2], xmin=[-2, -2], xmax=[2, 2], color="k")
        self.ax.vlines(x=[-2, 2], ymin=[-2, -2], ymax=[2, 2], color="k")

        X, Y = np.meshgrid(
            np.linspace(self.grid.min[0], self.grid.max[0], self.grid.pts_each_dim[0]),
            np.linspace(self.grid.min[1], self.grid.max[1], self.grid.pts_each_dim[1]),
            indexing="ij",
        )

        relative_state = self.relative_state(self.persuer_state, self.evader_state)
        index = self.grid.get_index(relative_state)
        angle = self.evader_state[2] % (2 * np.pi)
        Xr = X * np.cos(angle) - Y * np.sin(angle)
        Yr = X * np.sin(angle) + Y * np.cos(angle)

        self.ax.contour(
            Xr + self.evader_state[0],
            Yr + self.evader_state[1],
            self.brt[:, :, index[2]],
            levels=[0.1],
        )

        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(-2.5, 2.5)
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
        return (
            self.grid.get_value(
                self.brt, self.relative_state(self.persuer_state, self.evader_state)
            )
            < threshold
        )

    def opt_ctrl(self):
        # assert -np.pi <= self.evader_state[2] <= np.pi
        relative_state = self.relative_state(self.persuer_state, self.evader_state)
        index = self.grid.get_index(relative_state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        opt_ctrl = self.car.opt_ctrl_non_hcl(relative_state, spat_deriv)
        return opt_ctrl

    def opt_dstb(self):
        relative_state = self.relative_state(self.persuer_state, self.evader_state)
        index = self.grid.get_index(relative_state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        if spat_deriv[2] == 0:
            relative_state = self.relative_state2(self.persuer_state, self.evader_state)
            index = self.grid.get_index(relative_state)
            spat_deriv = spa_deriv(index, self.backup_brt, self.grid)
        opt_dstb = self.car.opt_dstb_non_hcl(spat_deriv)
        return opt_dstb

    def get_obs(self, evader_state, persuer_state, goal):
        return np.concatenate([self.theta_to_cos_sin(evader_state), 
                               self.theta_to_cos_sin(persuer_state), 
                               [np.linalg.norm(evader_state[:2] - persuer_state[:2])],
                               goal[:2]])

    def theta_to_cos_sin(self, state):
        return np.array(
            [state[0], state[1], np.cos(state[2]), np.sin(state[2])], dtype=np.float32
        )


if __name__ in "__main__":
    from datetime import datetime
    run_name = f"debug__{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"

    gym.logger.set_level(10)

    env = gym.make("Safe-Air3d-Fixed-Goal-v0")
    obs = env.reset()
    env.render()
    done = False
    while not done:
        if env.use_opt_ctrl():
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
