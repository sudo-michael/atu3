import numpy as np
from atu3.dynamics.air3d import Air3D
from odp.Grid.GridProcessing import Grid
from odp.dynamics import DubinsCar

grid = Grid(
    np.array([-2, -2, -np.pi]),
    np.array([2, 2, np.pi]),
    3,
    np.array([101, 101, 101]),
    [2],
)


car_r = 0.1
# NOTE ve != vp otherwise evader cannot excape persuer
# NOTE why is we_max different than wp_max?
#      think i'm just doing it for the sake of it, maybe remove it works fine
car_brt = Air3D(r=car_r, u_mode="max", d_mode="min", we_max=2.84, wp_max=2.00, ve=0.22, vp=0.15)
persuer_backup_brt = DubinsCar(x=[0, 0, 0], uMode='min', wMax=car_brt.wp_max, speed=car_brt.vp)
cylinder_r = car_r + car_r # could this be smaller?

VERSION=0

if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    ivf = CylinderShape(grid, [2], np.zeros(3), cylinder_r)

    def brt(version):
        lookback_length = 1.5
        t_step = 0.05
        small_number = 1e-5
        tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        compMethods = {"TargetSetMode": "minVWithV0"}

        car = car_brt

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

    def backup_brt(version):
        lookback_length = 5.0
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
        
