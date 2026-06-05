import threading
import time

import numpy as np
import pyrealsense2 as rs


class RealSenseWrapper:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intrinsics = color_profile.get_intrinsics()

        self.frame = None
        self.depth_image = None
        self.depth_frame = None
        self.break_flag = False
        self.lock = threading.Lock()

        self.get_frame_thread = threading.Thread(target=self.GetFrameThreadFunc)
        self.get_frame_thread.start()

    def __del__(self):
        try:
            self.StopThread()
        except Exception:
            pass

    def StopThread(self):
        if self.break_flag:
            return

        self.break_flag = True
        if self.get_frame_thread.is_alive():
            self.get_frame_thread.join()
        self.pipeline.stop()

    def GetFrameThreadFunc(self):
        while self.break_flag != True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                time.sleep(0.01)
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            with self.lock:
                self.frame = color_image
                self.depth_image = depth_image
                self.depth_frame = depth_frame

            time.sleep(0.01)

    def GetFrame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def GetDepthImage(self):
        with self.lock:
            if self.depth_image is None:
                return None
            return self.depth_image.copy()

    def GetDepthFrame(self):
        with self.lock:
            return self.depth_frame

    def GetColorIntrinsics(self):
        return self.color_intrinsics

    def GetDepthScale(self):
        return self.depth_scale
