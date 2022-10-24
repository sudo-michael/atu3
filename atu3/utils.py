import numpy as np
import gym
def normalize_angle(theta):
    """normalize theta to be in range (-pi, pi]"""
    print(f"{theta=}, deg: {theta / np.pi * 180}")
    return ((-theta + np.pi) % (2.0 * np.pi) - np.pi) * -1.0

class AutoResetWrapper(gym.Wrapper):
    # TODO cite from openai/gym
    """
    A class for providing an automatic reset functionality
    for gym environments when calling self.step().
    When calling step causes self.env.step() to return done,
    self.env.reset() is called,
    and the return format of self.step() is as follows:
    new_obs, terminal_reward, terminal_done, info
    new_obs is the first observation after calling self.env.reset(),
    terminal_reward is the reward after calling self.env.step(),
    prior to calling self.env.reset()
    terminal_done is always True
    info is a dict containing all the keys from the info dict returned by
    the call to self.env.reset(), with an additional key "terminal_observation"
    containing the observation returned by the last call to self.env.step()
    and "terminal_info" containing the info dict returned by the last call
    to self.env.step().
    If done is not true when self.env.step() is called, self.step() returns
    obs, reward, done, and info as normal.
    Warning: When using this wrapper to collect rollouts, note
    that the when self.env.step() returns done, a
    new observation from after calling self.env.reset() is returned
    by self.step() alongside the terminal reward and done state from the
    previous episode . If you need the terminal state from the previous
    episode, you need to retrieve it via the the "terminal_observation" key
    in the info dict. Make sure you know what you're doing if you
    use this wrapper!
    """

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if done:

            new_obs, new_info = self.env.reset()
            assert (
                "terminal_observation" not in new_info
            ), 'info dict cannot contain key "terminal_observation" '
            assert (
                "terminal_info" not in new_info
            ), 'info dict cannot contain key "terminal_info" '

            new_info["terminal_observation"] = obs
            new_info["terminal_info"] = info

            info = new_info

        return np.expand_dims(obs, axis=0), reward, done, info

def spa_deriv(index, V, g):
    """
    Calculates the spatial derivatives of V at an index for each dimension

    Args:
        index:
        V:
        g:

    Returns:
        List of left and right spatial derivatives for each dimension

    """
    spa_derivatives = []
    for dim, idx in enumerate(index):
        if dim == 0:
            left_index = []
        else:
            left_index = list(index[:dim])

        if dim == len(index) - 1:
            right_index = []
        else:
            right_index = list(index[dim + 1 :])

        next_index = tuple(left_index + [index[dim] + 1] + right_index)
        prev_index = tuple(left_index + [index[dim] - 1] + right_index)

        if idx == 0:
            if dim in g.pDim:
                left_periodic_boundary_index = tuple(
                    left_index + [V.shape[dim] - 1] + right_index
                )
                left_boundary = V[left_periodic_boundary_index]
            else:
                left_boundary = V[index] + np.abs(V[next_index] - V[index]) * np.sign(
                    V[index]
                )
            left_deriv = (V[index] - left_boundary) / g.dx[dim]
            right_deriv = (V[next_index] - V[index]) / g.dx[dim]
        elif idx == V.shape[dim] - 1:
            if dim in g.pDim:
                right_periodic_boundary_index = tuple(left_index + [0] + right_index)
                right_boundary = V[right_periodic_boundary_index]
            else:
                right_boundary = V[index] + np.abs(V[index] - V[prev_index]) * np.sign(
                    V[index]
                )
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (right_boundary - V[index]) / g.dx[dim]
        else:
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (V[next_index] - V[index]) / g.dx[dim]

        if dim==2:
            breakpoint()

        spa_derivatives.append((left_deriv + right_deriv) / 2)

    # print(spa_derivatives)
    # if isinstance(spa_derivatives[0], np.ndarray):
    #     breakpoint()
    return np.array(spa_derivatives)