import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from atu3.utils import normalize_angle, spa_deriv
from atu3.brt.brt_static_obstacle_3d import g as grid
from atu3.brt.brt_air_3d import car_brt
from atu3.brt.brt_static_obstacle_3d import goal_r
from atu3.deepreach.hji_1E2P import load_deepreach
from atu3.deepreach.dataio import xy_grid
import itertools
import jax.numpy as jnp


class Air3DNpEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self, render_mode=None, n=1, use_hj=False, deepreach_backend=False
    ) -> None:
        self.render_mode = render_mode
        self.car = car_brt
        # NOTE: let's try making the persuer slower since optimal disturbance makes it hard for the evader to escape
        self.car.vp = 0.21
        self.dt = 0.05
        self.use_hj = use_hj
        self.deepreach_backend = deepreach_backend
        if self.deepreach_backend:
            assert n == 2
        self.n = n

        self.action_space = gym.spaces.Box(
            low=-self.car.we_max, high=self.car.we_max, dtype=np.float32, shape=(1,)
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

        self.world_boundary = np.array([4.5, 4.5, np.pi], dtype=np.float32)

        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        path = os.path.abspath(__file__)
        if self.deepreach_backend:
            self.opt_ctrl_dstb_fn, self.value_fn, self.dataset_state = load_deepreach(
                "1e2p_atu3_6"
            )
        else:
            dir_path = os.path.dirname(path)
            self.brt = np.load(os.path.join(dir_path, f"assets/brts/air3d_brt_0.npy"))
            self.backup_brt = np.load(
                os.path.join(dir_path, f"assets/brts/backup_air3d_brt_0.npy")
            )
            self.grid = grid

    def step(self, action):
        info = {}
        info["used_hj"] = False

        if self.deepreach_backend:
            unnormalized_tcoords = self.deepreach_state(
                self.evader_state, self.persuer_states
            )
            value = self.value_fn(jnp.array(unnormalized_tcoords))
            print(f"Value: {value}")

        if self.use_hj and self.use_opt_ctrl():
            action = self.opt_ctrl()
            info["used_hj"] = True

        self.evader_state = (
            self.car.dynamics_non_hcl(0, self.evader_state, action) * self.dt
            + self.evader_state
        )
        self.evader_state[2] = normalize_angle(self.evader_state[2])

        if self.deepreach_backend:
            persuer_actions = self.opt_dstb(self.persuer_states)
            for i in range(self.n):
                self.persuer_states[i] = (
                    self.car.dynamics_non_hcl(
                        0, self.persuer_states[i], persuer_actions[i], is_evader=False
                    )
                    * self.dt
                    + self.persuer_states[i]
                )
                self.persuer_states[i][2] = normalize_angle(self.persuer_states[i][2])
        else:
            for i in range(self.n):
                persuer_action = self.opt_dstb(self.persuer_states[i])
                self.persuer_states[i] = (
                    self.car.dynamics_non_hcl(
                        0, self.persuer_states[i], persuer_action, is_evader=False
                    )
                    * self.dt
                    + self.persuer_states[i]
                )
                self.persuer_states[i][2] = normalize_angle(self.persuer_states[i][2])

        dist_to_goal = np.linalg.norm(self.evader_state[:2] - self.goal_location[:2])
        reward = -dist_to_goal
        terminated = False

        info["cost"] = 0

        if (
            np.linalg.norm(self.evader_state[:2] - self.goal_location)
            < self.goal_r + self.car.r
        ):
            terminated = True
            info["collision"] = "goal"
        # TODO: maybe change this to not be too close to a persuer instead of stopping on collision
        elif np.any(
            [
                np.linalg.norm(self.evader_state[:2] - self.persuer_states[i][:2])
                < self.car.r * 3
                for i in range(self.n)
            ]
        ):
            # terminated = True
            info['cost'] = 1.0
            info["collision"] = "persuer"

        if self.render_mode == "human":
            self.render()

        return (
            np.copy(
                self.get_obs(self.evader_state, self.persuer_states, self.goal_location)
            ),
            reward,
            terminated,
            False, # truncated
            info,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # self.evader_state = np.array([0.0, 0.0, np.pi / 2])
        # DEBUG: the following cases show when the brt is not working
        # works
        # self.evader_state = np.array([0.0, 0.0, 0.0])
        # self.persuer_states[0] = np.array([1.0, 0.3, -np.pi])
        # doesn't
        # self.persuer_states[1] = np.array([-1.0, -0.3, 0.0])
        # self.persuer_states[0] = np.array([0.5, -0.3, -np.pi])
        # self.persuer_states[0] = np.array([1.0, -0.0, -np.pi])
        # self.persuer_states[1] = np.array([-1.0, -0.0, 0.0])
        goal_locations = [
            np.array([1.5, 1.5]),
            np.array([0, 1.5]),
            np.array([-1.5, 1.5]),
            np.array([1.5, 0]),
            np.array([1.5, -1.5]),
        ]

        random_idx = np.random.randint(0, len(goal_locations))
        self.goal_location = goal_locations[random_idx]

        self.evader_state = np.random.uniform(
            low=-self.world_boundary, high=self.world_boundary
        )
        for i in range(self.n):
            self.persuer_states[i] = np.random.uniform(
                low=-self.world_boundary, high=self.world_boundary
            )

        # insert the persuer between the goal and the evader
        # self.persuer_states[i][:2] = goal_locations[random_idx] // 2

        info = {}
        info["cost"] = 0
        info["collision"] = "none"

        if self.render_mode == "human":
            self.render()

        return self.get_obs(self.evader_state, self.persuer_states, self.goal_location), info

    def render(self):
        if self.render_mode is None:
            return

        self.ax.clear()

        def add_robot(state, color="green"):
            self.ax.add_patch(plt.Circle(state[:2], radius=self.car.r, color=color))

            dir = state[:2] + self.car.r * np.array(
                [np.cos(state[2]), np.sin(state[2])]
            )

            self.ax.plot([state[0], dir[0]], [state[1], dir[1]], color="c")

        add_robot(self.evader_state, color="blue")
        for i in range(self.n):
            if i == 0:
                add_robot(self.persuer_states[i], color="red")
            else:
                add_robot(self.persuer_states[i], color="orange")
        goal = plt.Circle(self.goal_location[:2], radius=self.goal_r, color="g")
        self.ax.add_patch(goal)

        if not self.deepreach_backend:
            X, Y = np.meshgrid(
                np.linspace(
                    self.grid.min[0], self.grid.max[0], self.grid.pts_each_dim[0]
                ),
                np.linspace(
                    self.grid.min[1], self.grid.max[1], self.grid.pts_each_dim[1]
                ),
                indexing="ij",
            )

            relative_state = self.relative_state(self.persuer_states[0])
            index = self.grid.get_index(relative_state)
            angle = self.evader_state[2] % (2 * np.pi)
            Xr = X * np.cos(angle) - Y * np.sin(angle)
            Yr = X * np.sin(angle) + Y * np.cos(angle)

            # DEBUG: visualize relative state
            # add_robot(relative_state, color="orange")
            # add_robot(np.zeros(3), color="yellow")

            self.ax.contour(
                Xr + self.evader_state[0],
                Yr + self.evader_state[1],
                # X, Y,
                self.brt[:, :, index[2]],
                levels=[0.1],
            )

        if self.deepreach_backend:
            # Get the meshgrid in the (x, y) coordinate
            grid_points = 200
            mgrid_coords = xy_grid(
                200,
                x_max=self.dataset_state.alpha["x"],
                y_max=self.dataset_state.alpha["y"],
            )
            ones = jnp.ones((mgrid_coords.shape[0], 1))
            unnormalized_tcoords = jnp.concatenate(
                (
                    self.dataset_state.t_max * ones,
                    mgrid_coords,
                    self.persuer_states[0][0] * ones,
                    self.persuer_states[0][1] * ones,
                    self.persuer_states[1][0] * ones,
                    self.persuer_states[1][1] * ones,
                    self.evader_state[2] * ones,
                    self.persuer_states[0][2] * ones,
                    self.persuer_states[1][2] * ones,
                ),
                axis=1,
            )
            V = self.value_fn(unnormalized_tcoords)

            V = np.array(V)
            V = V.reshape((grid_points, grid_points))

            # unnormalize value function
            V = (
                V * self.dataset_state.var
            ) / self.dataset_state.norm_to + self.dataset_state.mean

            V = (V <= 0.001) * 1.0

            X, Y = np.meshgrid(
                np.linspace(
                    -self.dataset_state.alpha["x"],
                    self.dataset_state.alpha["x"],
                    grid_points,
                ),
                np.linspace(
                    -self.dataset_state.alpha["y"],
                    self.dataset_state.alpha["y"],
                    grid_points,
                ),
                indexing="ij",
            )

            self.ax.contour(
                X,
                Y,
                V,
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

        if self.render_mode == "human":
            self.fig.canvas.flush_events()
            plt.pause(1 / self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return img

    def close(self):
        plt.close()
        return

    def use_opt_ctrl(self, threshold=0.001):
        if self.deepreach_backend:
            unnormalized_tcoords = self.deepreach_state(
                self.evader_state, self.persuer_states
            )
            value = self.value_fn(jnp.array(unnormalized_tcoords))
            return value.item() < threshold
        else:
            relative_state = self.relative_state(self.persuer_states[0])
            return self.grid.get_value(self.brt, relative_state) < threshold

    def opt_ctrl(self):
        if self.deepreach_backend:
            unnormalized_tcoords = self.deepreach_state(
                self.evader_state, self.persuer_states
            )
            opt_ctrl, _ = self.opt_ctrl_dstb_fn(
                jnp.array(unnormalized_tcoords)
            )  # (1, ), _
            return np.array(opt_ctrl[0])  # (1, )
        # elif self.n > 1:
        #     raise NotImplementedError("Only support 1 persuer for now")
        else:
            relative_state = self.relative_state(self.persuer_states[0])
            index = self.grid.get_index(relative_state)
            spat_deriv = spa_deriv(index, self.brt, self.grid)
            opt_ctrl = self.car.opt_ctrl_non_hcl(relative_state, spat_deriv)
            return opt_ctrl

    def opt_dstb(self, persuer_state):
        if self.deepreach_backend:
            unnormalized_tcoords = self.deepreach_state(
                self.evader_state, self.persuer_states
            )
            _, opt_dstbs = self.opt_ctrl_dstb_fn(jnp.array(unnormalized_tcoords))
            return np.array(opt_dstbs)
        else:
            relative_state = self.relative_state(persuer_state)
            index = self.grid.get_index(relative_state)
            spat_deriv = spa_deriv(index, self.brt, self.grid)
            if spat_deriv[2] == 0:
                relative_state = persuer_state - self.evader_state
                relative_state[2] = persuer_state[2]
                index = self.grid.get_index(relative_state)
                spat_deriv = spa_deriv(index, self.backup_brt, self.grid)

            opt_dstb = self.car.opt_dstb_non_hcl(spat_deriv)
            return opt_dstb

    def relative_state(self, persuer_state):
        rotated_relative_state = np.zeros(3)
        relative_state = persuer_state - self.evader_state

        angle = -self.evader_state[2]

        # fmt: off
        # brt assume that evader_state is at theta=0
        rotated_relative_state[0] = relative_state[0] * np.cos(angle) - relative_state[1] * np.sin(angle)
        rotated_relative_state[1] = relative_state[0] * np.sin(angle) + relative_state[1] * np.cos(angle)
        # fmt: on

        # after rotating by -evader_state[2], the relative angle will still be the same
        rotated_relative_state[2] = normalize_angle(relative_state[2])
        # print(rotated_relative_state)
        return rotated_relative_state

    def get_obs(self, evader_state, persuer_states, goal):
        # return [x y cos(theta) sin(theta) all evader and persuers states]
        t = (
            [self.theta_to_cos_sin(evader_state)],
            list(map(self.theta_to_cos_sin, persuer_states)),
            [goal[:2]],
        )
        return np.concatenate((tuple(itertools.chain.from_iterable(t)))).astype(np.float32)

    def deepreach_state(self, evader_state, persuer_states):
        def state_to_unnormalized_tcoords(evader_state, persuer_states, t):
            # [state sequence: 0, 1,   2,   3     4,    5,    6,    7,       8,        9].
            # [state sequence: t, x_e, y_e, x_p1, y_p1, x_p2, y_p2, theta_e, theta_p1, theta_p2].
            unnormalized_tcoords = jnp.array(
                [
                    t,
                    evader_state[0],
                    evader_state[1],
                    persuer_states[0][0],
                    persuer_states[0][1],
                    persuer_states[1][0],
                    persuer_states[1][1],
                    evader_state[2],
                    persuer_states[0][2],
                    persuer_states[1][2],
                ]
            )
            return unnormalized_tcoords

        state = jnp.expand_dims(
            state_to_unnormalized_tcoords(
                evader_state, persuer_states, t=self.dataset_state.t_max
            ),
            0,
        )
        return state

    def theta_to_cos_sin(self, state):
        return np.array(
            [state[0], state[1], np.cos(state[2]), np.sin(state[2])], dtype=np.float32
        )


if __name__ in "__main__":
    import atu3
    from datetime import datetime

    run_name = f"debug__{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"

    gym.logger.set_level(10)

    # env = Air3DNpEnv(2, use_hj=True, deepreach_backend=True)
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
    # def make_env():
    #     def thunk():
    #         return gym.make('Safe-Air9D-v0', n=2, use_hj=True, deepreach_backend=True)

    #     return thunk
    # envs = gym.vector.SyncVectorEnv([make_env()])
    # breakpoint()
    # import wandb
    # wandb.init(monitor_gym=True)
    env = gym.make("Safe-Air3D-v0", render_mode='rgb_array')
    # env = gym.wrappers.RecordVideo(env, video_folder="./videos/")
    # env = gym.make("CartPole-v1")
    # env = Air3DNpEnv(1, use_hj=False, deepreach_backend=False)
    breakpoint()
    obs, info = env.reset()
    done = False
    t = 0
    for t in range(int(1e5)):
        action = env.action_space.sample()
        # if t % 2 == 0:
        #     action = np.array([env.unwrapped.car.we_max])
        # else:
        #     action = np.array([-env.unwrapped.car.we_max])
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        if done:
            done = False
            print(info)
            exit()
            obs, info = env.reset()

    env.close()
