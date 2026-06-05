import importlib
import os
import sys
import types
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def install_dependency_stubs():
    if "cv2" not in sys.modules:
        cv2_stub = types.ModuleType("cv2")
        cv2_stub.FONT_HERSHEY_PLAIN = 1
        cv2_stub.putText = lambda frame, *args, **kwargs: frame
        cv2_stub.rectangle = lambda frame, *args, **kwargs: frame
        cv2_stub.circle = lambda frame, *args, **kwargs: frame
        cv2_stub.line = lambda frame, *args, **kwargs: frame
        cv2_stub.UMat = lambda frame: frame
        sys.modules["cv2"] = cv2_stub

    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.YOLO = lambda *args, **kwargs: None
    sys.modules["ultralytics"] = ultralytics_stub

    rclpy_stub = types.ModuleType("rclpy")
    rclpy_stub.ok = lambda: True
    rclpy_stub.init = lambda: None
    sys.modules["rclpy"] = rclpy_stub

    rclpy_node_stub = types.ModuleType("rclpy.node")

    class Node:
        pass

    rclpy_node_stub.Node = Node
    sys.modules["rclpy.node"] = rclpy_node_stub

    geometry_msgs_stub = types.ModuleType("geometry_msgs")
    geometry_msgs_msg_stub = types.ModuleType("geometry_msgs.msg")

    class Twist:
        def __init__(self):
            self.linear = types.SimpleNamespace(x=0.0)
            self.angular = types.SimpleNamespace(z=0.0)

    geometry_msgs_msg_stub.Twist = Twist
    sys.modules["geometry_msgs"] = geometry_msgs_stub
    sys.modules["geometry_msgs.msg"] = geometry_msgs_msg_stub


install_dependency_stubs()
robot_module = importlib.import_module("RobotController.RobotController")
RobotController = robot_module.RobotController


class FakeIntrinsics:
    ppx = 50.0
    ppy = 40.0
    fx = 50.0
    fy = 50.0


class FakeDepthFrame:
    def __init__(self, obstacle_distance=0.8):
        self.obstacle_distance = obstacle_distance

    def get_distance(self, x, y):
        if 38 <= x <= 62 and 28 <= y <= 68:
            return self.obstacle_distance
        return 2.5


class DepthByXFrame:
    def get_distance(self, x, y):
        if 40 <= x <= 60:
            return 1.0
        return 2.6


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBox:
    def __init__(self, x1, y1, x2, y2, track_id=None):
        self.xyxy = [[
            FakeScalar(x1),
            FakeScalar(y1),
            FakeScalar(x2),
            FakeScalar(y2),
        ]]
        self.id = None if track_id is None else FakeScalar(track_id)


def make_controller():
    controller = object.__new__(RobotController)
    controller.target_id = 1
    controller.id_str = ""
    controller.is_tracking = False
    controller.last_linear_velocity = 0.0
    controller.person_path = []
    controller.lost_search_start_time = None
    controller.lost_search_direction = 1
    controller.lost_search_last_switch_time = None
    controller.target_history = []
    controller.last_target_lost_time = None
    controller.last_image_search_direction = 1
    controller.track_state = "lost_stop"
    controller.predicted_target_odom = None
    controller.prediction_confidence = 0.0
    controller.last_grid_info = None
    controller.last_seen_odom = None
    controller.last_seen_time = None
    controller.last_seen_arrived = False
    controller.branch_queue = []
    controller.current_branch = None
    controller.branch_start_time = None
    controller.branch_start_odom = None
    controller.branch_phase = None
    controller.branch_plan_initialized = False
    controller.last_reacquire_time = None
    controller.last_reacquire_score = None
    controller.intent_processor = None
    controller.intent_thumb_candidate_id = None
    controller.intent_thumb_confirmations = 0
    controller.intent_last_result = None
    controller.intent_last_action = "idle"
    controller.fps_counter = types.SimpleNamespace(Count=lambda: None, GetFps=lambda: 0.0)
    controller.ros2_transfer = types.SimpleNamespace(SendCmdVel=lambda linear, angular: None)
    return controller


class RobotControllerLogicTest(unittest.TestCase):
    def test_standby_intent_requires_consecutive_thumb_before_tracking(self):
        controller = make_controller()

        first_result = types.SimpleNamespace(
            person_id=5,
            gesture_label=robot_module.TRACKING_ENTER_GESTURE,
            request_tag="standby",
        )
        second_result = types.SimpleNamespace(
            person_id=5,
            gesture_label=robot_module.TRACKING_ENTER_GESTURE,
            request_tag="standby",
        )

        self.assertFalse(controller.HandleStandbyGestureResult(first_result))
        self.assertFalse(controller.GetIsTracking())

        self.assertTrue(controller.HandleStandbyGestureResult(second_result))
        self.assertTrue(controller.GetIsTracking())
        self.assertEqual(controller.GetTargetId(), 5)

    def test_tracking_intent_palm_resets_tracking_state(self):
        controller = make_controller()
        controller.is_tracking = True
        controller.target_id = 7
        controller.person_path = [(1.0, 0.0)]

        result = types.SimpleNamespace(
            person_id=7,
            gesture_label=robot_module.TRACKING_EXIT_GESTURE,
            request_tag="tracking:7",
        )

        self.assertTrue(controller.HandleTrackingGestureResult(result))
        self.assertFalse(controller.GetIsTracking())
        self.assertEqual(controller.GetTargetId(), robot_module.kDefaultTrackId)
        self.assertEqual(controller.person_path, [])

    def test_depth_grid_marks_obstacle_and_inflates(self):
        controller = make_controller()
        grid_info = controller.BuildLocalGrid(FakeDepthFrame(0.8), FakeIntrinsics(), 100, 80)

        self.assertTrue(grid_info["valid"])
        self.assertEqual(
            controller.GetLocalGridCellStatus(grid_info, 0.8, 0.0),
            robot_module.kGridOccupied,
        )
        self.assertNotEqual(
            controller.GetLocalGridCellStatus(grid_info, 0.65, 0.0),
            robot_module.kGridFree,
        )

    def test_safety_limits_stop_forward_motion_for_blocked_grid(self):
        controller = make_controller()
        grid_info = controller.BuildLocalGrid(FakeDepthFrame(0.8), FakeIntrinsics(), 100, 80)
        obstacle_info = {
            "front_distance": None,
            "left_distance": 2.0,
            "right_distance": 0.8,
            "front_region": None,
            "left_region": None,
            "right_region": None,
            "active": False,
            "mode": "clear",
        }

        linear, angular, obstacle_info = controller.ApplySafetyLimits(0.6, 0.0, obstacle_info, grid_info)

        self.assertEqual(linear, 0.0)
        self.assertNotEqual(angular, 0.0)
        self.assertTrue(obstacle_info["active"])

    def test_close_front_obstacle_always_stops_forward_motion(self):
        controller = make_controller()
        grid_info = controller.CreateLocalGridInfo()
        obstacle_info = {
            "front_distance": 0.3,
            "left_distance": 2.0,
            "right_distance": 0.8,
            "front_region": None,
            "left_region": None,
            "right_region": None,
            "active": False,
            "mode": "clear",
        }

        linear, angular, obstacle_info = controller.ApplySafetyLimits(0.2, 0.0, obstacle_info, grid_info)

        self.assertEqual(linear, 0.0)
        self.assertNotEqual(angular, 0.0)
        self.assertEqual(obstacle_info["mode"], "stop_turn")

    def test_fuzzy_control_drives_forward_for_far_centered_target(self):
        controller = make_controller()
        obstacle_info = {
            "front_distance": 2.0,
            "left_distance": 2.0,
            "right_distance": 2.0,
            "front_region": None,
            "left_region": None,
            "right_region": None,
            "active": False,
            "mode": "clear",
        }

        command, fuzzy_info = controller.CalculateFuzzyTargetCommand(0.7, 0.0, obstacle_info, 0.6)

        self.assertIsNotNone(command)
        self.assertGreater(command[2], 0.0)
        self.assertAlmostEqual(command[1], 0.0)
        self.assertEqual(fuzzy_info["distance_label"], "far")
        self.assertEqual(fuzzy_info["obstacle_label"], "clear")

    def test_fuzzy_control_stops_forward_motion_for_danger_obstacle(self):
        controller = make_controller()
        obstacle_info = {
            "front_distance": 0.3,
            "left_distance": 2.0,
            "right_distance": 2.0,
            "front_region": None,
            "left_region": None,
            "right_region": None,
            "active": False,
            "mode": "clear",
        }

        command, fuzzy_info = controller.CalculateFuzzyTargetCommand(0.7, 0.0, obstacle_info, 0.6)

        self.assertIsNotNone(command)
        self.assertEqual(command[0], 0.0)
        self.assertEqual(command[2], 0.0)
        self.assertEqual(fuzzy_info["obstacle_label"], "danger")

    def test_prediction_uses_odom_velocity(self):
        controller = make_controller()
        original_time = robot_module.time.time
        robot_module.time.time = lambda: 10.0
        try:
            controller.target_history = [
                {"time": 8.0, "person_odom": (0.0, 0.0)},
                {"time": 9.0, "person_odom": (1.0, 0.0)},
            ]
            controller.last_target_lost_time = 9.5

            predicted, confidence, lost_age = controller.PredictLostTargetOdom((0.0, 0.0, 0.0), ("/odom", "qos", 1, 0.1))
        finally:
            robot_module.time.time = original_time

        self.assertAlmostEqual(lost_age, 0.5)
        self.assertGreater(confidence, 0.0)
        self.assertAlmostEqual(predicted[0], 2.0)
        self.assertAlmostEqual(predicted[1], 0.0)

    def test_odom_invalid_allows_only_short_blind_motion(self):
        controller = make_controller()

        short_command = controller.CalculateBlindImageSearchCommand(0.8)
        long_command = controller.CalculateBlindImageSearchCommand(1.2)

        self.assertIsNotNone(short_command)
        self.assertEqual(short_command[0], robot_module.kOdomBlindMaxForwardVelocity)
        self.assertIsNone(long_command)

    def test_visible_target_updates_last_seen_and_disappear_direction(self):
        controller = make_controller()

        controller.UpdateTargetObservation(
            FakeBox(70, 20, 90, 60),
            (80, 40),
            1.2,
            (0.1, 0.0, 1.2),
            (2.0, 3.0),
            100,
            80,
        )

        self.assertEqual(controller.last_seen_odom, (2.0, 3.0))
        self.assertEqual(controller.last_image_search_direction, -1)
        self.assertFalse(controller.last_seen_arrived)

    def test_last_seen_arrival_threshold(self):
        controller = make_controller()
        controller.last_seen_odom = (1.0, 0.0)

        self.assertTrue(controller.IsLastSeenReached((0.6, 0.0, 0.0)))
        self.assertFalse(controller.IsLastSeenReached((0.0, 0.0, 0.0)))

    def test_branch_queue_prefers_disappear_direction_then_other_open_branches(self):
        controller = make_controller()
        controller.last_image_search_direction = -1
        grid_info = controller.CreateLocalGridInfo()
        grid_info["valid"] = True
        for distance in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for offset in [-0.2, 0.0, 0.2]:
                controller.MarkLocalGridCell(grid_info, distance, offset, robot_module.kGridFree)
                controller.MarkLocalGridCell(grid_info, offset, -distance, robot_module.kGridFree)
                controller.MarkLocalGridCell(grid_info, offset, distance, robot_module.kGridFree)

        branch_queue = controller.BuildBranchQueue(grid_info)

        self.assertEqual(branch_queue[0]["name"], "right")
        self.assertEqual({branch["name"] for branch in branch_queue}, {"left", "front", "right"})

    def test_branch_explore_limits_distance_and_time(self):
        controller = make_controller()
        controller.current_branch = {"name": "front", "heading": 0.0, "search_direction": 1}
        controller.branch_phase = "forward"
        controller.branch_start_odom = (0.0, 0.0, 0.0)
        controller.branch_start_time = 10.0
        original_time = robot_module.time.time
        robot_module.time.time = lambda: 11.0
        try:
            command = controller.CalculateBranchExploreCommand((0.5, 0.0, 0.0))
            self.assertIsNotNone(command)
            self.assertEqual(command[0], robot_module.kBranchForwardVelocity)

            done_command = controller.CalculateBranchExploreCommand((1.1, 0.0, 0.0))
            self.assertIsNone(done_command)
            self.assertIsNone(controller.current_branch)
        finally:
            robot_module.time.time = original_time

    def test_lost_target_plans_to_last_seen_before_prediction(self):
        controller = make_controller()
        controller.is_tracking = True
        controller.last_seen_odom = (1.0, 0.0)
        controller.target_history = [
            {"time": 8.0, "person_odom": (2.0, 0.0)},
            {"time": 9.0, "person_odom": (4.0, 0.0)},
        ]
        controller.last_target_lost_time = 9.5
        planned_targets = []

        def fake_plan(target_robot, grid_info, max_forward_velocity):
            planned_targets.append(target_robot)
            return (0.1, 0.0, 0.1), {"target_forward": target_robot[0], "target_lateral": target_robot[1]}

        controller.PlanLocalCommandToTarget = fake_plan
        original_time = robot_module.time.time
        robot_module.time.time = lambda: 10.0
        try:
            controller.TrackAndDraw(
                np.zeros((80, 100, 3), dtype=np.uint8),
                None,
                FakeDepthFrame(2.5),
                FakeIntrinsics(),
                (0.0, 0.0, 0.0),
                "/odom",
                ("/odom", "qos", 1, 0.1),
                -1,
            )
        finally:
            robot_module.time.time = original_time

        self.assertGreater(len(planned_targets), 0)
        self.assertEqual(planned_targets[0], (1.0, 0.0))

        controller.current_branch = {"name": "front", "heading": 0.0, "search_direction": 1}
        controller.branch_phase = "forward"
        controller.branch_start_odom = (0.0, 0.0, 0.0)
        controller.branch_start_time = 10.0
        robot_module.time.time = lambda: 14.1
        try:
            timeout_command = controller.CalculateBranchExploreCommand((0.1, 0.0, 0.0))
            self.assertIsNone(timeout_command)
            self.assertIsNone(controller.current_branch)
        finally:
            robot_module.time.time = original_time

    def test_reacquire_updates_target_id_by_last_seen_odom(self):
        controller = make_controller()
        controller.is_tracking = True
        controller.target_id = 1
        controller.last_seen_odom = (1.0, 0.0)
        controller.target_history = [
            {
                "time": 9.0,
                "center_x": 50,
                "center_y": 40,
                "center_norm_x": 0.5,
                "center_norm_y": 0.5,
                "box_width": 10,
                "box_height": 40,
                "distance": 1.0,
                "person_point": (0.0, 0.0, 1.0),
                "person_odom": (1.0, 0.0),
            }
        ]

        matched_box = controller.ReacquireTarget(
            [
                FakeBox(75, 20, 85, 60, track_id=8),
                FakeBox(45, 20, 55, 60, track_id=7),
            ],
            DepthByXFrame(),
            FakeIntrinsics(),
            (0.0, 0.0, 0.0),
            100,
            80,
        )

        self.assertIsNotNone(matched_box)
        self.assertEqual(controller.GetBoxTrackId(matched_box), 7)
        self.assertEqual(controller.GetTargetId(), 7)

    def test_reacquire_rejects_far_candidate(self):
        controller = make_controller()
        controller.is_tracking = True
        controller.target_id = 1
        controller.last_seen_odom = (1.0, 0.0)
        controller.target_history = [
            {
                "time": 9.0,
                "center_x": 50,
                "center_y": 40,
                "center_norm_x": 0.5,
                "center_norm_y": 0.5,
                "box_width": 10,
                "box_height": 40,
                "distance": 1.0,
                "person_point": (0.0, 0.0, 1.0),
                "person_odom": (1.0, 0.0),
            }
        ]

        matched_box = controller.ReacquireTarget(
            [FakeBox(75, 20, 85, 60, track_id=8)],
            DepthByXFrame(),
            FakeIntrinsics(),
            (0.0, 0.0, 0.0),
            100,
            80,
        )

        self.assertIsNone(matched_box)
        self.assertEqual(controller.GetTargetId(), 1)


if __name__ == "__main__":
    unittest.main()
