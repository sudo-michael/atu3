import gym
import numpy as np
import matplotlib.pyplot as plt
from atu.utils import normalize_angle


class Air3dEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self) -> None:
        # state
        self.evader_state = np.array([1, 1, -1])
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

    def step(self):
        pass

    def reset(self, seed=None):
        pass

    def render(self, mode="human"):
        self.ax.clear()
        def add_robot(state, color='green'):
            self.ax.add_patch(
                plt.Circle(state[:2], radius=self.car.r, color=color)
            )

            dir = state[:2] + self.car.r * np.array(
                [np.cos(state[2]), np.sin(state[2])]
            )

            self.ax.plot([state[0], dir[0]], [state[1], dir[1]], color="c")

        add_robot(self.evader_state, color='blue')
        add_robot(self.persuer_state, color='red')
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
            plt.pause(1 / self.metadata["render_fps"])
        return img

    def close(self):
        plt.close()
        return


if __name__ in "__main__":
    import atu
    env = gym.make('Safe-Air3d-v0')
    obs = env.reset()
    env.render()