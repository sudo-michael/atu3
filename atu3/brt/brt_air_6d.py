import numpy as np
import odp
from atu3.dynamics.air6d import Air6D
from odp.Grid.GridProcessing import Grid

grid = Grid(
    np.array([-2, -2, -np.pi, -2, -2, -np.pi]),
    np.array([2, 2, np.pi, 2, 2, np.pi]),
    6,
    np.array([35, 35, 16, 35, 35, 16]),
    # np.array([40, 40, 24, 40, 40, 24]) // 10,
    np.array([2, 5]),
)

car_r = 0.1

car_brt = Air6D(r=car_r, u_mode="max", d_mode="min", we_max=2.84, wp_max=2.84, ve=0.22, vp=0.12)
car_brt_2 = Air6D(r=car_r, u_mode="max", d_mode="min", we_max=1.5, wp_max=1.0, ve=1.0, vp=0.5)

cylinder_r = car_r + car_r

if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    X1, X2, X3, X4, X5, X6 = np.meshgrid(*grid.grid_points, indexing='ij')

    def collide(X1, X2, X4, X5):
        return np.sqrt((X1 - X4)**2 + (X2 - X5)**2) - cylinder_r

    def wall(X4, X5):
        def lower_half_space(x, value):
            return x - value
        
        def upper_half_space(x, value):
            return -x + value

        return np.minimum(
            np.minimum(lower_half_space(X4, -2.0 + car_r), upper_half_space(X4, 2.0 - car_r)),
            np.minimum(lower_half_space(X5, -2.0 + car_r), upper_half_space(X5, 2.0 - car_r))
        )

    # ivf = np.minimum(collide(X1, X2, X4, X5), wall(X4, X5))
    ivf = collide(X1, X2, X4, X5)


    # def brt(version):
    #     lookback_length = 3.0
    #     t_step = 0.06
    #     small_number = 1e-5
    #     tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    #     compMethods = {"TargetSetMode": "minVWithV0"}

    #     if version == 1:
    #         car = car_brt
    #     elif version == 2:
    #         car = car_brt_2

    #     result = HJSolver(
    #         car,
    #         grid,
    #         ivf, 
    #         tau,
    #         compMethods,
    #         PlotOptions(
    #             do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[]
    #         ),
    #         saveAllTimeSteps=False,
    #     )

    #     np.save(f"./atu3/envs/assets/brts/air6d_brt_no_wall_5_40_v{version}.npy", result)

    # brt(VERSION)

    def brt():
        lookback_length = 3.0
        t_step = 0.06
        small_number = 1e-5
        tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        compMethods = {"TargetSetMode": "minVWithV0"}

        result = HJSolver(
            car,
            grid,
            ivf, 
            tau,
            compMethods,
            PlotOptions(
                do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[]
            ),
            saveAllTimeSteps=False,
        )

        np.save(f"./atu3/envs/assets/brts/air6d_brt_no_wall_5_40_v{version}.npy", result)
