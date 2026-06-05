
import cv2
import signal

from RealSenseWrapper import RealSenseWrapper
from OdomWrapper import OdomWrapper
from RobotController import RobotController

break_flag = False
def set_break_flag(signum, frame):
    print("set_break_flag")
    global break_flag
    break_flag = True

signal.signal(signal.SIGINT, set_break_flag)

realsense_wrapper = RealSenseWrapper.RealSenseWrapper()
robot_controller = RobotController.RobotController()
odom_wrapper = OdomWrapper.OdomWrapper()

while break_flag != True:
    key = cv2.waitKey(1)
    if key == 27:
        break

    ############
    # get frame
    ############
    bgr_frame = realsense_wrapper.GetFrame()
    if type(bgr_frame) == type(None):
        continue
    depth_frame = realsense_wrapper.GetDepthFrame()
    color_intrinsics = realsense_wrapper.GetColorIntrinsics()
    odom_pose = odom_wrapper.GetPose()
    odom_topic = odom_wrapper.GetTopic()
    odom_status = odom_wrapper.GetStatus()

    ####################
    # control the robot
    ####################
    final_frame = robot_controller.Run(bgr_frame, depth_frame, color_intrinsics, odom_pose, odom_topic, odom_status, key)
    if type(final_frame) == type(None):
        continue

    #################
    # show final img
    #################
    cv2.imshow("DR People Tracking", final_frame)

realsense_wrapper.StopThread()
odom_wrapper.StopThread()

cv2.destroyAllWindows()
