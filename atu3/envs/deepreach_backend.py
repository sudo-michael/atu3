from matplotlib import pyplot as plt
from atu3.deepreach_verification.dynamics.dynamics import MultiVehicleCollision
from atu3.deepreach_verification.utils.modules import SingleBVPNet
import os
import numpy as np
import torch




class DeepReachBackend:
    x_resolution=200
    y_resolution=200
    z_resolution=5
    time_resolution=3

    def __init__(self, experiment_dir='atu_multivehiclecollisionavoidtube_2') -> None:
        self.dynamics = MultiVehicleCollision(diff_model=True)
        self.model = SingleBVPNet(
            in_features=self.dynamics.input_dim,
            out_features=1,
            type="sine",
            mode="mlp",
            final_layer_factor=1.0,
            hidden_features=512,  # NOTE: hardcode for now
            num_hidden_layers=3,  # NOTE: hardcode for now
        ).cuda()

        deepreach_verifcation_path = os.path.join(os.path.dirname(__file__), "../deepreach_verification/runs/")

        model_path = os.path.join(
            deepreach_verifcation_path, experiment_dir, "training", "checkpoints", "model_final.pth"
        )
        self.model.load_state_dict(torch.load(model_path))

        # state: [time x_e, y_e, x_p1, y_p1, x_p2, y_p2, theta_e, theta_p1, theta_p2]

    def V(self, state):
        model_results = self.model(
            {"coords": self.dynamics.coord_to_input(state.cuda())}
        )
        values = self.dynamics.io_to_value(
            model_results["model_in"].detach(),
            model_results["model_out"].squeeze(dim=-1).detach(),
        )
        return values.item() # scalar

    def opt_ctrl_dstb(self, state):
        model_results = self.model(
            {"coords": self.dynamics.coord_to_input(state.cuda())}
        )
        dvs = self.dynamics.io_to_dv(
            model_results["model_in"],
            model_results["model_out"].squeeze(dim=-1),
        )

        opt_ctrl = self.dynamics.optimal_control(
            state[:, 1:].cuda(), dvs[..., 1:].cuda()
        )

        opt_dstb = self.dynamics.optimal_disturbance(
            state[:, 1:].cuda(), dvs[..., 1:].cuda()
        )

        return opt_ctrl.detach().cpu(), opt_dstb.detach().cpu() # (1, 1), # (1, 2)

    def ham(self, state, action):
        model_results = self.model(
            {"coords": self.dynamics.coord_to_input(state.cuda())}
        )
        dvs = self.dynamics.io_to_dv(
            model_results["model_in"],
            model_results["model_out"].squeeze(dim=-1),
        )

        ham = self.dynamics.agent_hamiltonian(state[:, 1:].cuda(), dvs[..., 1:].cuda(), action)

        return ham.item() # scalar

    def plot(self):
        plot_config = self.dynamics.plot_config()

        state_test_range = self.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        # NOTE: hardcode
        t_max = 1.0
        times = torch.linspace(0, t_max, self.time_resolution)
        xs = torch.linspace(x_min, x_max, self.x_resolution)
        ys = torch.linspace(y_min, y_max, self.y_resolution)
        zs = torch.linspace(z_min, z_max, self.z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        
        fig = plt.figure(figsize=(5*len(times), 5*len(zs)))
        for i in range(len(times)):
            for j in range(len(zs)):
                coords = torch.zeros(self.x_resolution*self.y_resolution, self.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

                with torch.no_grad():
                    model_results = self.model({'coords': self.dynamics.coord_to_input(coords.cuda())})
                    values = self.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
                
                ax = fig.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                ax.set_title('t = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))
                s = ax.imshow(1*(values.detach().cpu().numpy().reshape(self.x_resolution, self.y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                fig.colorbar(s) 
        plt.show()


# def generate_trajectory():
# backend = DeepReachBackend('atu_multivehiclecollisionavoidtube_2')
# backend.plot()

# fmt:off
# state = torch.tensor([[1.0, 
#                        0.0, 0.0, 
#                        0.5, 0.0,  # p1
#                        0.0, -0.5,  # p2
#                        0, 
#                        -np.pi, 
#                        np.pi/2]]).cuda()
# # fmt:on
# def is_captured(state, capture_radius=0.2):
#     if torch.norm(state[..., 1:3] - state[..., 3:5], dim=-1).item() < capture_radius:
#         print("p1 capture")
#         return True
#     elif torch.norm(state[..., 1:3] - state[..., 5:7], dim=-1).item() < capture_radius:
#         print("p2 capture")
#         return True
#     return False

# x_traj = [state[0][1].item()]
# y_traj = [state[0][2].item()]

# x_p1_traj = [state[0][3].item()]
# y_p1_traj = [state[0][4].item()]

# x_p2_traj = [state[0][5].item()]
# y_p2_traj = [state[0][6].item()]

# for _ in range(200):
#     opt_ctrl, opt_dstb = backend.opt_ctrl_dstb(state) # (b, 1), # (b, 2)
#     dsdt = torch.roll(backend.dynamics.dsdt(state, opt_ctrl, opt_dstb), 1, -1)
#     # state[0, 3:] = state[0, 3:] + dsdt[0, 3:] * 0.05
#     state = state + dsdt * 0.05
#     print(f"{backend.V(state)}")
#     print(f"x:{state[0][1]:2f} y:{state[0][2]:2f} xp1:{state[0][3]:2f} yp1:{state[0][4]:2f} xp2:{state[0][5]:2f} yp2:{state[0][6]:2f}")

#     x_traj.append(state[0][1].item())
#     y_traj.append(state[0][2].item())
#     x_p1_traj.append(state[0][3].item())
#     y_p1_traj.append(state[0][4].item())
#     x_p2_traj.append(state[0][5].item())
#     y_p2_traj.append(state[0][6].item())
#     # state = backend.dynamics.
#     if is_captured(state):
        

#         print('capture')
#         break

# plt.plot(x_traj, y_traj, label='ego')
# plt.plot(x_p1_traj, y_p1_traj, label='p1')
# plt.plot(x_p2_traj, y_p2_traj, label='p2', linestyle='--')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.savefig('traj')