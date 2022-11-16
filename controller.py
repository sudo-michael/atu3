import rospy
import tf
import numpy as np
from geometry_msgs.msg import TransformStamped
from jetracer.msg import jetRacerCar as JetRacerCarMsg
from collections import namedtuple
import math
import time
from threading import Lock
import subprocess

VICON_CAR_TOPIC = "vicon/jetracer_1/jetracer_1"
TURTLEBOT_CMD_VEL_TOPIC = ""

DEBUG = False

class Controller:
    def __init__(self):
        rospy.init_node("tb3_controller_node")
        # rospy.loginfo("starting jetracer controller node...")
        # rospy.loginfo("using value function: {}".format(self.V))

        rospy.loginfo(f"starting subscriber for {VICON_CAR_TOPIC}")
        rospy.Subscriber(VICON_CAR_TOPIC, TransformStamped, self.callback, queue_size=1)

        self.publisher = rospy.Publisher(
            TURTLEBOT_CMD_VEL_TOPIC, cmd_vel?, queue_size=1
        )

        self.boundary_epsilon = 0.15

        # play a sound when optimal control takes over
        self.play_sound = True

        while not rospy.is_shutdown():
            rospy.spin()



    def in_bound(self, state):
        return (-3.0 <= state[0] <= 3.0) and (-1.0 <= state[1] <= 4.0)

    def callback(self, ts_msg):
        pose = ts_msg.transform
        position = self.Position(x=pose.translation.x, y=pose.translation.y)
        # velocity = self.calculate_velocity(position)
        theta = TODO
        state = (pose.translation..x, pose.translation.y, theta)

        rospy.logdebug(
            "car's location on the grid\n \
            x: {} y: {}, v: {}, theta: {}".format(
            self.grid._Grid__grid_points[0][i],
            self.grid._Grid__grid_points[1][j],
            self.grid._Grid__grid_points[2][k],
            self.grid._Grid__grid_points[3][l]))


        value = self.grid.get_value(self.V, state)

        # if we near the bonudary, allow optimal control to take over for 0.5 seconds
        # before handing control back to user
        # near the bonudary of BRT, apply optimal control
        if value <= self.boundary_epsilon and self.in_bound(state):
            opt_a, opt_w = self.car.opt_ctrl(state, (0, 0, dV_dx3, dV_dx4))
            # rospy.loginfo("opt_a: {} opt_w: {}".format(opt_a, opt_w))

            # if self.play_sound:
            #     subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",  "/home/michael/catkin_ws/src/jetracer_controller/scripts/safe.wav"])
            #     self.play_sound = False

            #     rospy.logwarn("optimal control taking over!")
            #     if jetracer_msg.steerAngle < 0:
            #         rospy.loginfo("throttle: {} steerAngle: {} {}".format(jetracer_msg.throttle, jetracer_msg.steerAngle, "left"))
            #     else:
            #         rospy.loginfo("throttle: {} steerAngle: {} {}".format(jetracer_msg.throttle, jetracer_msg.steerAngle, "right"))
            #     rospy.loginfo("value: {}".format(value))
            #     rospy.loginfo("in grid world\nx: {} y: {}, v: {}, theta: {}".format(
            #         self.grid._Grid__grid_points[0][i],
            #         self.grid._Grid__grid_points[1][j],
            #         self.grid._Grid__grid_points[2][k],
            #         self.grid._Grid__grid_points[3][l]))

            #     rospy.loginfo("irl\n x: {} y: {}, v: {}, theta: {}".format(
            #         state[0], state[1], state[2], state[3]))

            # rospy.loginfo("throttle: {} steerAngle: {}".format(jetracer_msg.throttle, jetracer_msg.steerAngle))
            self.publisher.publish(self.optimal_msg)

def main():
    try:
        # rospy.logwarn(
        #     "Remember to check that grid and car function are the same as in user_definer.py"
        # )
        controller = Controller()
    except rospy.ROSInterruptException:
        rospy.loginfo("shutdown")
        pass


if __name__ == "__main__":
    main()