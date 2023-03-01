import numpy as np
from atu3.dynamics.DubinsCar import DubinsCar
from odp.Grid.GridProcessing import Grid

g = Grid(
    np.array([-5, -5, -np.pi]),
    np.array([5, 5, np.pi]),
    3,
    np.array([101, 101, 101]),
    [2],
)
car_r = 0.5
car_brt = DubinsCar(r=car_r, uMode="max", dMode="min", wMax=1.5)
cylinder_r = car_r + 0.1 # cylinder is another car, 0.1 for some extra space

if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    ivf = CylinderShape(g, [2], np.zeros(3), cylinder_r)

    def brt(d=True):
        lookback_length = 5.0
        t_step = 0.05
        small_number = 1e-5
        tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        compMethods = {"TargetSetMode": "minVWithV0"}

        result = HJSolver(
            car_brt,
            g,
            ivf, 
            tau,
            compMethods,
            PlotOptions(
                do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[]
            ),
            saveAllTimeSteps=False,
        )

        np.save("./atu3/envs/assets/brts/static_obstacle_brt.npy", result)

    brt(d=False)