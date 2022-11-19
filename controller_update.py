import rospy
import numpy as np
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Twist
import subprocess

from scipy.spatial.transform import Rotation as R
from atu3.brt.brt_static_obstacle_3d import g, car_brt
from atu3.brt.brt_air_3d import grid
from atu3.brt.brt_air_3d import car_brt as air_3d_car_brt
from atu3.utils import spa_deriv, normalize_angle
VICON_CAR_TOPIC = "vicon/turtlebot_1/turtlebot_1"
TURTLEBOT_CMD_VEL_TOPIC = "cmd_vel"

DEBUG = False

THETA_OFFSET = np.pi / 2

class Controller:
    def __init__(self, id, vicon_topic, tb_cmd_vel_topic, type="pursuer"):
        self.type = type
        self.id = id
        self.vicon_topic = vicon_topic
        self.tb_cmd_vel_topic = tb_cmd_vel_topic
        rospy.init_node(f"tb3_{self.type}_ctrl_node_{self.id}")
        # rospy.loginfo("starting jetracer controller node...")
        # rospy.loginfo("using value function: {}".format(self.V))

        rospy.loginfo(f"starting subscriber for {self.vicon_topic}")
        print('starting up')
        rospy.Subscriber(self.vicon_topic, TransformStamped, self.callback, queue_size=1)
        self.publisher = rospy.Publisher(self.tb_cmd_vel_topic, Twist, queue_size=1)
        
        # TODO set rate of sublisher
        if self.type == "pursuer":
            version = "1"
            self.brt = np.load(f"./atu3/envs/assets/brts/air3d_brt_{version}.npy")
            self.backup_brt = np.load(f"./atu3/envs/assets/brts/backup_air3d_brt_{version}.npy")
            self.grid = grid
            self.car = air_3d_car_brt
        elif self.type == "evader":    
            self.grid = g
            self.car = car_brt
            self.brt = np.load("./atu3/envs/assets/brts/static_obstacle_brt.npy")
        self.boundary_epsilon = 0.15
        # play a sound when optimal control takes over
        self.play_sound = True

        while not rospy.is_shutdown():
            rospy.spin()


    def in_bound(self, state):
        return True
        # return (-3.0 <= state[0] <= 3.0) and (-1.0 <= state[1] <= 4.0)

    def callback(self, ts_msg):
        pose = ts_msg.transform
        x = ts_msg.transform.rotation.x
        y = ts_msg.transform.rotation.y
        z = ts_msg.transform.rotation.z
        w = ts_msg.transform.rotation.w
        rot = R.from_quat([x, y, z, w])
        radians = rot.as_euler('xyz', degrees=False)  # seq order; degrees - True-degees/False-radians
        theta = normalize_angle(radians[2] - THETA_OFFSET)
        state = (pose.translation.x, pose.translation.y, theta)
        rospy.loginfo(f"x: {pose.translation.x}, y: {pose.translation.y}, theta: {theta}")

        print(f"{theta=}")
        index = self.grid.get_index(state)
        i, j, k = index
        rospy.logdebug(
            "car's location on the grid\n \
            x: {} y: {}, theta: {}".format(
            self.grid.grid_points[0][i],
            self.grid.grid_points[1][j],
            self.grid.grid_points[2][k],))


        value = self.grid.get_value(self.brt, state)

        spat_deriv = spa_deriv(index, self.brt, self.grid)
        opt_ctrl = self.car.opt_ctrl_non_hcl(0, state, spat_deriv)
        if np.linalg.norm(np.array([state[0], state[1]])) < 0.2:
            vel_msg = Twist()
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
        else:
            vel_msg = Twist()
            vel_msg.linear.x = 0.22
            vel_msg.angular.z = opt_ctrl[0]
        self.publisher.publish(vel_msg)

def main():
    try:
        # rospy.logwarn(
        #     "Remember to check that grid and car function are the same as in user_definer.py"
        # )
        print('main')
        # 1. make 2 Controllers() that publies cmd_vel to different turtlebots
            # 1 should be for the pursuer
            # 1 should be for the evader
        pursuer_ctrl = Controller(id="turtlebot_1", vicon_topic="vicon/turtlebot_1/turtlebot_1", tb_cmd_vel_topic="cmd_vel", type="pursuer")
        evader_ctrl = Controller(id="turtlebot_2", vicon_topic="vicon/turtlebot_1/turtlebot_2", tb_cmd_vel_topic="cmd_vel", type="evader")
        
        # 2. have a way to log the x, y, theta position of each persuer and evader 
        # 3. for the persuer's controller, it should be using opt_dstb like in env/air_3d.py
            # this is probably easiset
            # use version 1 brts
            # self.brt = np.load(os.path.join(dir_path, f"assets/brts/air3d_brt_{version}.npy"))
            # self.backup_brt = np.load(os.path.join(dir_path, f"assets/brts/backup_air3d_brt_{version}.npy"))
            # instad of using grid and brt_car from brt_static_obstacle, take them from brt_air_3d
            # verion=1
            # 1. fake where evade_state is
        # 4. for the evader's controller, it should actor(obs) like in the sac.py
            # get_obs from air_3d.py
            # copy the actor class from sac.py
            # load saved model from models
            # goal is at [2.5, 2.5] in the real world
        # if there's a collision (too close) or reaches goal, send stop to all turtlebots
    except rospy.ROSInterruptException:
        rospy.loginfo("shutdown")
        pass
    
if __name__ == "__main__":
    main()
