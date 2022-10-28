import numpy as np
import odp
from atu3.dynamics.air3d import Air3D
from odp.Grid.GridProcessing import Grid
from odp.dynamics import DubinsCar

grid = Grid(
    np.array([-3, -3, -np.pi]),
    np.array([3, 3, np.pi]),
    3,
    np.array([101, 101, 101]),
    [2],
)

car_r = 0.2
car_brt = Air3D(r=car_r, u_mode="max", d_mode="min", we_max=1.5, wp_max=1.5, ve=1.0, vp=0.7)

persuer_backup_brt = DubinsCar(x=[0, 0, 0], uMode='min', wMax=car_brt.wp_max, speed=car_brt.ve)

cylinder_r = car_r + car_r + 0.2

if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    ivf = CylinderShape(grid, [2], np.zeros(3), cylinder_r)

    def brt(d=True):
        lookback_length = 2.0
        t_step = 0.05
        small_number = 1e-5
        tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        compMethods = {"TargetSetMode": "minVWithV0"}

        result = HJSolver(
            car_brt,
            grid,
            ivf, 
            tau,
            compMethods,
            PlotOptions(
                do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[]
            ),
            saveAllTimeSteps=False,
        )

        np.save("./atu3/envs/assets/brts/air3d_brt.npy", result)

    brt(d=False)

    def backup_brt():
        lookback_length = 2.0
        t_step = 0.05
        small_number = 1e-5
        tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        compMethods = {"TargetSetMode": "minVWithV0"}

        result = HJSolver(
            persuer_backup_brt,
            grid,
            ivf, 
            tau,
            compMethods,
            PlotOptions(
                do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[]
            ),
            saveAllTimeSteps=False,
        )

        np.save("./atu3/envs/assets/brts/backup_air3d_brt.npy", result)

    brt(d=False)
    backup_brt()