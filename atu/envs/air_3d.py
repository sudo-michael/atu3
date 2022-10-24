import gym
import numpy as np
import matplotlib.pyplot as plt
from atu.utils import normalize_angle


class DubinsCar:
    def __init__(
        self, we_max=1, wp_max=1, ve=1, vp=1, r=0.2, u_mode="max", d_mode="min"
    ) -> None:
        self.ve = ve
        self.vp = vp
        self.we_max = we_max
        self.wp_max = wp_max
        self.r = r

        self.u_mode = u_mode
        self.d_mode = d_mode

    def dynamics_non_hcl(self, t, state, ctrl, is_evader=True):
        v = self.ve if is_evader else self.vp
        x_dot = v * np.cos(state[2])
        y_dot = v * np.sin(state[2])
        theta_dot = ctrl[0]
        return np.array([x_dot, y_dot, theta_dot], dtype=np.float32)


class Air3dEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self) -> None:
        self.dynamics = DubinsCar()
        # state
        self.evader_state = np.array([1, 1, 0])
        self.persuer_state = np.array([-1, -1, 0])
        self.goal_location = np.array([2, 2, 0.2])
        # world
        self.world_width = 10
        self.world_height = 10
        self.left_wall = -4.5
        self.right_wall = 4.5
        self.bottom_wall = -4.5
        self.top_wall = 4.5

        self.fig, self.ax = plt.subplots(figsize=(5, 5))

    def step(self, action):
        self.evader_state = (
            self.dynamics.dynamics_non_hcl(0, self.evader_state, action, is_evader=True)
            * self.dt
            + self.evader_state
        )
        self.persuer_state = (
            self.dynamics.dynamics_non_hcl(0, self.persuer_state, -action, is_evader=False)
            * self.dt
            + self.persuer_state
        )

        done = False

        if (
            np.linalg.norm(
                self.relative_state(self.persuer_state, self.evader_state)[:2]
            )
            <= 0.5
        ):
            done = True

        return (
            {
                "persuer_state": np.copy(self.persuer_state),
                "evader_state": np.copy(self.evader_state),
                "goal_location": np.copy(self.goal_location),
            },
            0,
            done,
            {},
        )

    def reset(self, seed=None):
        def dynamics_to_obs(state):
            return np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2])])

        # TODO
        return None

    def render(self, mode="human"):
        self.ax.clear()

        def add_robot(state, color="green"):
            self.ax.add_patch(plt.Circle(state[:2], radius=self.dynamics.r, color=color))

            dir = state[:2] + self.dynamics.r * np.array(
                [np.cos(state[2]), np.sin(state[2])]
            )

            self.ax.plot([state[0], dir[0]], [state[1], dir[1]], color="c")

        add_robot(self.evader_state, color="blue")
        add_robot(self.persuer_state, color="red")
        goal = plt.Circle(
            self.goal_location[:2], radius=self.goal_location[2], color="g"
        )
        self.ax.add_patch(goal)

        # walls
        self.ax.hlines(y=[-4.5, 4.5], xmin=[-4.5, -4.5], xmax=[4.5, 4.5], color="k")
        self.ax.vlines(x=[-4.5, 4.5], ymin=[-4.5, -4.5], ymax=[4.5, 4.5], color="k")

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
            # plt.pause(1 / self.metadata["render_fps"])
            plt.show()
        return img

    def close(self):
        plt.close()
        return

    def relative_state(self, persuer_state, evader_state):
        relative_state = persuer_state - evader_state
        relative_state[2] = normalize_angle(relative_state[2])
        return relative_state


if __name__ in "__main__":
    import atu

    env = gym.make("Safe-Air3d-v0")
    obs = env.reset()
    env.render()

