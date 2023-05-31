import numpy as np
from atu3.dynamics.DubinsCar import DynDubinsCar
from odp.Grid.GridProcessing import Grid

g = Grid(
    np.array([-4, -4, -np.pi]),
    np.array([4, 4, np.pi]),
    3,
    np.array([101, 101, 101]),
    [2],
)
car_r = 0.5
car_brt = DynDubinsCar(r=car_r, uMode="max", dMode="min", speed=0.22, wMax=2.84)
car_brat = DynDubinsCar(r=car_r, uMode="min", dMode="max", speed=0.22, wMax=2.84)
obstacle_r = 0.5
goal_r = 0.2
cylinder_r = car_r + obstacle_r

if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    ivf = CylinderShape(g, [2], np.zeros(3), cylinder_r)
    goal = CylinderShape(g, [2], np.array([1.5, 1.5, 0]), goal_r)

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
                do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[]
            ),
            saveAllTimeSteps=False,
        )

        np.save("./atu3/envs/assets/brts/static_obstacle_brt.npy", result)

    # brt(d=False)
    def brat(d=True):
        # I think there's something wrong with the brat solver
        # when lookback_length is too long
        lookback_length = 15.0
        t_step = 0.05
        small_number = 1e-5
        tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        compMethods = {"TargetSetMode": "minVWithVTarget", "ObstacleSetMode": "maxVWithObstacle"} 

        result = HJSolver(
            car_brat,
            g,
            [goal, ivf], 
            tau,
            compMethods,
            PlotOptions(
                do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[]
            ),
            saveAllTimeSteps=False,
        )

        np.save("./atu3/envs/assets/brts/static_obstacle_brat.npy", result)
    # brat(d=False)

    def min_brt(d=True):
        lookback_length = 15.0
        t_step = 0.05
        small_number = 1e-5
        tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        compMethods = {"TargetSetMode": "minVWithV0"} 

        result = HJSolver(
            car_brat,
            g,
            goal, 
            tau,
            compMethods,
            PlotOptions(
                do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[]
            ),
            saveAllTimeSteps=False,
        )
        np.save("./atu3/envs/assets/brts/static_obstacle_min_brt.npy", result)
    min_brt()