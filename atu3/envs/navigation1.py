# environment from https://github.com/abalakrishna123/recovery-rl/blob/master/env/navigation1.py
import gymnasium as gym
import numpy as np

START_POS = [-50, 0]
END_POS = [0, 0]
GOAL_THRESH = 1.0
START_STATE = START_POS
GOAL_STATE = END_POS

MAX_FORCE = 1
HORIZON = 100

NOISE_SCALE = 0.05
AIR_RESIST = 0.2

HARD_MODE = False

OBSTACLE = [[[-100, 150], [5, 10]], [[-100, -80], [-10, 10]], [[-100, 150], [-10, -5]]]

CAUTION_ZONE = [[[-100, 150], [4, 5]], [[-100, 150], [-5, -4]]]


class Navigation1(gym.Env):
    def __init__(self):
        self.hist = self.cost = self.done = self.time = self.state = None
        self.A = np.eye(2)
        self.B = np.eye(2)

        self.action_space = gym.spaces.Box(
            -np.ones(2) * MAX_FORCE, np.ones(2) * MAX_FORCE
        )
        self.observation_space = gym.spaces.Box(
            -np.ones(2) * np.float("inf"), np.ones(2) * np.float("inf")
        )
        self.obstacle = OBSTACLE
        self.caution_zone = CAUTION_ZONE
        self.goal = GOAL_STATE

    def step(self, a):
        a = np.clip(a, -MAX_FORCE, MAX_FORCE)
        old_state = self.state.copy()
        next_state = self._next_state(self.state, a)
        cur_cost = self.step_cost(self.state, a)
        self.cost.append(cur_cost)
        self.state = next_state
        self.time += 1
        self.hist.append(self.state)
        self.done = cur_cost > -4 or self.obstacle(next_state)

        return (
            self.state,
            cur_cost,
            self.done,
            {
                "constraint": self.obstacle(next_state),
                "reward": cur_cost,
                "state": old_state,
                "next_state": next_state,
                "action": a,
                "success": cur_cost > -4,
            },
        )

    def reset(self):
        self.state = START_STATE + np.random.randn(2)
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        return self.state

    def _next_state(self, s, a, override=False):
        if self.obstacle(s):
            print("obs", s)
            return s
        return self.A.dot(s) + self.B.dot(a) + NOISE_SCALE * np.random.randn(len(s))

    def step_cost(self, s, a):
        if HARD_MODE:
            return -int(np.linalg.norm(np.subtract(GOAL_STATE, s)) < GOAL_THRESH)
        return -np.linalg.norm(np.subtract(GOAL_STATE, s))