import numpy as np
import odp
from atu3.dynamics.air3d import Air3D
from odp.Grid.GridProcessing import Grid
from odp.dynamics import DubinsCar

grid = Grid(
    np.array([-1, -1, -np.pi]),
    np.array([1, 1, np.pi]),
    3,
    np.array([101, 101, 101]),
    [2],
)

car_r = 0.1
# NOTE ve != vp otherwise evader cannot excape persuer
car_brt = Air3D(r=car_r, u_mode="max", d_mode="min", we_max=2.00, wp_max=2.84, ve=0.22, vp=0.14)
car_brt_2 = Air3D(r=car_r, u_mode="max", d_mode="min", we_max=2.00, wp_max=2.84, ve=0.22, vp=0.14)
persuer_backup_brt = DubinsCar(x=[0, 0, 0], uMode='min', wMax=car_brt.wp_max, speed=car_brt.vp)

VERSION=3
cylinder_r = car_r + car_r

if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    ivf = CylinderShape(grid, [2], np.zeros(3), cylinder_r)

    def brt(version=1):
        lookback_length = 1.5
        t_step = 0.05
        small_number = 1e-5
        tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        compMethods = {"TargetSetMode": "minVWithV0"}

        if version == 3:
            car = car_brt
        elif version == 2:
            car = car_brt_2

        result = HJSolver(
            car,
            grid,
            ivf, 
            tau,
            compMethods,
            PlotOptions(
                do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[]
            ),
            saveAllTimeSteps=False,
        )

        np.save(f"./atu3/envs/assets/brts/air3d_brt_{version}.npy", result)

    def backup_brt(version=1):
        lookback_length = 1.5
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

        np.save(f"./atu3/envs/assets/brts/backup_air3d_brt_{version}.npy", result)

    brt(VERSION)
    backup_brt(VERSION)
        
