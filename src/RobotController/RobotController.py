
import cv2
import math
import numpy as np
import time

from . FpsCounter import FpsCounter
from . YoloWrapper import YoloWrapper
from . ROSTransfer import TransferConstants

kUseRos1Transfer = False

if kUseRos1Transfer:
    from . ROSTransfer import ROS1Transfer
else:
    from . ROSTransfer import ROS2Transfer

kDefaultTrackId = 0
kDefaultTrackMode = False
kTargetDistance = 0.5
kDistanceDeadband = 0.05
kLinearDistanceKp = 0.6
kMaxForwardVelocity = 0.6
kLinearVelocitySmooth = 0.6
kAngularHeadingKp = 1.2
kHeadingDeadband = 0.05
kPathMinPointDistance = 0.10
kPathLookaheadDistance = 0.60
kPathTargetDeadband = 0.08
kPathTurnInPlaceHeading = 0.45
kLostPathMaxForwardVelocity = 0.35
kMaxPathPoints = 1000
kObstacleStopDistance = 0.40
kObstacleSlowDistance = 0.80
kObstacleTurnVelocity = 0.45
kSearchRadianVelocity = 0.45
kSearchSwitchInterval = 1.6
kSearchMaxDuration = 8.0
kRobotSafetyRadius = 0.35
kLocalGridForwardMin = 0.0
kLocalGridForwardMax = 3.0
kLocalGridLateralMin = -1.5
kLocalGridLateralMax = 1.5
kLocalGridResolution = 0.10
kDepthGridPixelStep = 24
kTargetHistoryMax = 12
kPredictionMinHorizon = 1.0
kPredictionMaxHorizon = 2.0
kPredictionMaxDuration = 4.0
kPredictionMinConfidence = 0.20
kOdomValidMaxAge = 0.80
kOdomBlindMoveDuration = 1.0
kOdomBlindMaxForwardVelocity = 0.15
kPlannerSimTime = 1.2
kPlannerSimStep = 0.2
kPlannerTargetDeadband = 0.15
kLastSeenArriveDistance = 0.5
kBranchExploreMaxDistance = 1.0
kBranchExploreMaxDuration = 3.0
kBranchTurnHeadingTolerance = 0.25
kBranchForwardVelocity = 0.18
kBranchTurnVelocity = 0.45
kReacquireMaxOdomDistance = 1.2
kReacquireMaxImageDistance = 0.35
kReacquireMaxLostDuration = 8.0
kGridUnknown = -1
kGridFree = 0
kGridOccupied = 1
kGridInflated = 2

class RobotController(object):
    def __init__(self):
        self.fps_counter = FpsCounter.FpsCounter()
        self.yolo_wrapper = YoloWrapper.YoloWrapper()
        if kUseRos1Transfer:
            self.ros1_transfer = ROS1Transfer.ROS1Transfer()
        else:
            self.ros2_transfer = ROS2Transfer.ROS2Transfer()
    
        self.is_tracking = kDefaultTrackMode
        self.target_id = kDefaultTrackId
        self.id_str = ""

        self.last_linear_velocity = 0.0
        self.person_path = []
        self.lost_search_start_time = None
        self.lost_search_direction = 1
        self.lost_search_last_switch_time = None
        self.target_history = []
        self.last_target_lost_time = None
        self.last_image_search_direction = 1
        self.track_state = "lost_stop"
        self.predicted_target_odom = None
        self.prediction_confidence = 0.0
        self.last_grid_info = None
        self.last_seen_odom = None
        self.last_seen_time = None
        self.last_seen_arrived = False
        self.branch_queue = []
        self.current_branch = None
        self.branch_start_time = None
        self.branch_start_odom = None
        self.branch_phase = None
        self.branch_plan_initialized = False
        self.last_reacquire_time = None
        self.last_reacquire_score = None

    def SetTargetId(self, id):
        self.target_id = id

    def GetTargetId(self):
        return self.target_id

    def SetIsTracking(self, state):
        self.is_tracking = state

    def GetIsTracking(self):
        return self.is_tracking

    def GetBoxTrackId(self, box):
        if box is None or not hasattr(box, "id") or box.id is None:
            return None
        return int(box.id.item())

    def GetBoxBounds(self, box):
        return (
            int(box.xyxy[0][0].item()),
            int(box.xyxy[0][1].item()),
            int(box.xyxy[0][2].item()),
            int(box.xyxy[0][3].item()),
        )

    def FindTarget(self, boxes):
        for box in boxes:
            track_id = self.GetBoxTrackId(box)
            if track_id == self.GetTargetId():
                return box

    def GetLastObservation(self):
        if len(self.target_history) == 0:
            return None
        return self.target_history[-1]

    def GetReacquireReferenceOdom(self):
        if self.last_seen_odom is not None:
            return self.last_seen_odom
        if self.predicted_target_odom is not None:
            return self.predicted_target_odom
        valid_points = [item for item in self.target_history if item["person_odom"] is not None]
        if len(valid_points) == 0:
            return None
        return valid_points[-1]["person_odom"]

    def ScoreReacquireCandidate(self, box, depth_frame, color_intrinsics, odom_pose, image_width, image_height, allow_image_fallback=True):
        track_id = self.GetBoxTrackId(box)
        if track_id is None or track_id == self.GetTargetId():
            return None

        x1, y1, x2, y2 = self.GetBoxBounds(box)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        person_distance, _ = self.GetBoxCenterDistance(
            depth_frame, x1, y1, x2, y2, image_width, image_height)
        person_point = self.DeprojectPixelToPoint(
            color_intrinsics, center[0], center[1], person_distance)

        person_odom = None
        if person_point is not None:
            person_lateral = -person_point[0]
            person_forward = person_point[2]
            person_odom = self.TransformPersonToOdom(person_forward, person_lateral, odom_pose)

        last_observation = self.GetLastObservation()
        image_score = None
        if last_observation is not None:
            center_norm_x = float(center[0]) / max(1, image_width)
            center_norm_y = float(center[1]) / max(1, image_height)
            dx = center_norm_x - last_observation["center_norm_x"]
            dy = center_norm_y - last_observation["center_norm_y"]
            image_score = math.sqrt(dx * dx + dy * dy)

        reference_odom = self.GetReacquireReferenceOdom()
        odom_score = None
        if reference_odom is not None and person_odom is not None:
            dx = person_odom[0] - reference_odom[0]
            dy = person_odom[1] - reference_odom[1]
            odom_score = math.sqrt(dx * dx + dy * dy)

        distance_score = 0.0
        size_score = 0.0
        if last_observation is not None:
            if person_distance is not None and last_observation["distance"] is not None:
                distance_score = min(1.0, abs(person_distance - last_observation["distance"]) / 2.0)
            last_width = max(1, last_observation["box_width"])
            last_height = max(1, last_observation["box_height"])
            width_ratio = abs((max(1, x2 - x1) - last_width) / float(last_width))
            height_ratio = abs((max(1, y2 - y1) - last_height) / float(last_height))
            size_score = min(1.0, 0.5 * (width_ratio + height_ratio))

        if odom_score is not None:
            if odom_score > kReacquireMaxOdomDistance:
                return None
            score = odom_score
            if image_score is not None:
                score += 0.25 * image_score
            score += 0.15 * distance_score + 0.10 * size_score
            return score

        if not allow_image_fallback or image_score is None or image_score > kReacquireMaxImageDistance:
            return None
        return image_score + 0.15 * distance_score + 0.10 * size_score

    def ReacquireTarget(self, boxes, depth_frame, color_intrinsics, odom_pose, image_width, image_height):
        if len(self.target_history) == 0 and self.last_seen_odom is None and self.predicted_target_odom is None:
            return None

        allow_image_fallback = True
        if self.last_target_lost_time is not None:
            lost_age = time.time() - self.last_target_lost_time
            if lost_age > kReacquireMaxLostDuration:
                allow_image_fallback = False

        best_box = None
        best_score = None
        for box in boxes:
            score = self.ScoreReacquireCandidate(
                box, depth_frame, color_intrinsics, odom_pose, image_width, image_height,
                allow_image_fallback)
            if score is None:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_box = box

        if best_box is None:
            return None

        self.SetTargetId(self.GetBoxTrackId(best_box))
        self.last_reacquire_time = time.time()
        self.last_reacquire_score = best_score
        return best_box

    def GetBoxCenterDistance(self, depth_frame, x1, y1, x2, y2, image_width, image_height):
        if depth_frame is None:
            return None, None

        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width - 1, x2))
        y2 = max(0, min(image_height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None, None

        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        half_width = max(3, box_width // 8)
        half_height = max(3, box_height // 8)

        rx1 = max(0, center_x - half_width)
        ry1 = max(0, center_y - half_height)
        rx2 = min(image_width - 1, center_x + half_width)
        ry2 = min(image_height - 1, center_y + half_height)

        distances = []
        step_x = max(1, (rx2 - rx1) // 8)
        step_y = max(1, (ry2 - ry1) // 8)

        for y in range(ry1, ry2 + 1, step_y):
            for x in range(rx1, rx2 + 1, step_x):
                distance = depth_frame.get_distance(x, y)
                if distance > 0:
                    distances.append(distance)

        if len(distances) == 0:
            return None, (rx1, ry1, rx2, ry2)

        return float(np.median(distances)), (rx1, ry1, rx2, ry2)

    def GetDepthRegionDistance(self, depth_frame, image_width, image_height, x_min_ratio, x_max_ratio, y_min_ratio, y_max_ratio):
        if depth_frame is None:
            return None, None

        x1 = int(image_width * x_min_ratio)
        x2 = int(image_width * x_max_ratio)
        y1 = int(image_height * y_min_ratio)
        y2 = int(image_height * y_max_ratio)
        x1 = max(0, min(image_width - 1, x1))
        x2 = max(0, min(image_width - 1, x2))
        y1 = max(0, min(image_height - 1, y1))
        y2 = max(0, min(image_height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None, None

        distances = []
        step_x = max(1, (x2 - x1) // 10)
        step_y = max(1, (y2 - y1) // 8)
        for y in range(y1, y2 + 1, step_y):
            for x in range(x1, x2 + 1, step_x):
                distance = depth_frame.get_distance(x, y)
                if distance > 0:
                    distances.append(distance)

        if len(distances) == 0:
            return None, (x1, y1, x2, y2)
        return float(np.median(distances)), (x1, y1, x2, y2)

    def GetObstacleInfo(self, depth_frame, image_width, image_height):
        front_distance, front_region = self.GetDepthRegionDistance(
            depth_frame, image_width, image_height, 0.35, 0.65, 0.45, 0.85)
        left_distance, left_region = self.GetDepthRegionDistance(
            depth_frame, image_width, image_height, 0.05, 0.35, 0.45, 0.85)
        right_distance, right_region = self.GetDepthRegionDistance(
            depth_frame, image_width, image_height, 0.65, 0.95, 0.45, 0.85)

        return {
            "front_distance": front_distance,
            "left_distance": left_distance,
            "right_distance": right_distance,
            "front_region": front_region,
            "left_region": left_region,
            "right_region": right_region,
            "active": False,
            "mode": "clear",
        }

    def ApplyObstacleAvoidance(self, linear_velocity, radian_velocity, obstacle_info):
        front_distance = obstacle_info["front_distance"]
        if front_distance is None or linear_velocity <= 0.0:
            return linear_velocity, radian_velocity, obstacle_info

        if front_distance <= kObstacleStopDistance:
            left_distance = obstacle_info["left_distance"]
            right_distance = obstacle_info["right_distance"]
            if left_distance is not None and right_distance is not None:
                turn_direction = 1 if left_distance > right_distance else -1
            else:
                turn_direction = -1

            obstacle_info["active"] = True
            obstacle_info["mode"] = "stop_turn"
            return 0.0, turn_direction * kObstacleTurnVelocity, obstacle_info

        if front_distance <= kObstacleSlowDistance:
            scale = (front_distance - kObstacleStopDistance) / (kObstacleSlowDistance - kObstacleStopDistance)
            scale = max(0.0, min(1.0, scale))
            obstacle_info["active"] = True
            obstacle_info["mode"] = "slow"
            return linear_velocity * scale, radian_velocity, obstacle_info

        return linear_velocity, radian_velocity, obstacle_info

    def DrawObstacleInfo(self, frame, obstacle_info, y):
        if obstacle_info is None:
            return

        for region, color in [
            (obstacle_info["left_region"], [128, 255, 255]),
            (obstacle_info["front_region"], [0, 255, 255]),
            (obstacle_info["right_region"], [128, 255, 255]),
        ]:
            if region is not None:
                x1, y1, x2, y2 = region
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        front_distance = obstacle_info["front_distance"]
        if front_distance is None:
            distance_text = "--"
        else:
            distance_text = "{:.2f}".format(front_distance)
        cv2.putText(frame,"obs {} front {}".format(obstacle_info["mode"], distance_text),
                    (0,y), cv2.FONT_HERSHEY_PLAIN, 1.2, [0,128,255], 1)

    def DrawPlanningInfo(self, frame, grid_info, planner_info, target_robot, image_width, y):
        target_text = "--"
        if target_robot is not None:
            target_text = "x {:.2f} z {:.2f}".format(target_robot[1], target_robot[0])
        nearest = "--"
        if grid_info is not None and grid_info["nearest_obstacle"] is not None:
            nearest = "{:.2f}".format(grid_info["nearest_obstacle"])
        branch_text = "--"
        if self.current_branch is not None:
            branch_text = "{} {}".format(self.current_branch["name"], self.branch_phase)
        elif len(self.branch_queue) > 0:
            branch_text = "queue {}".format(len(self.branch_queue))
        cv2.putText(frame,"state {} conf {:.2f} target {} near {} branch {}".format(
                    self.track_state, self.prediction_confidence, target_text, nearest, branch_text),
                    (0,y), cv2.FONT_HERSHEY_PLAIN, 1.2, [0,128,255], 1)

        if target_robot is not None:
            center = (image_width // 2, 455)
            target_forward, target_lateral = target_robot
            endpoint = (
                int(center[0] + max(-80, min(80, target_lateral * 55))),
                int(center[1] - max(-80, min(80, target_forward * 35))),
            )
            cv2.line(frame, center, endpoint, [0, 255, 255], 2)
            cv2.circle(frame, endpoint, 4, [0, 255, 255], -1)

        if grid_info is None or not grid_info["valid"]:
            return

        cell_size = 3
        origin_x = max(0, image_width - grid_info["cols"] * cell_size - 8)
        origin_y = 20
        for row in range(0, grid_info["rows"], 1):
            for col in range(0, grid_info["cols"], 1):
                status = int(grid_info["grid"][row, col])
                if status == kGridUnknown:
                    continue
                if status == kGridFree:
                    color = [80, 80, 80]
                elif status == kGridOccupied:
                    color = [0, 0, 255]
                else:
                    color = [0, 160, 255]
                x1 = origin_x + col * cell_size
                y1 = origin_y + (grid_info["rows"] - 1 - row) * cell_size
                cv2.rectangle(frame, (x1, y1), (x1 + cell_size, y1 + cell_size), color, -1)

        robot_col = int((0.0 - kLocalGridLateralMin) / kLocalGridResolution)
        robot_x = origin_x + robot_col * cell_size
        robot_y = origin_y + (grid_info["rows"] - 1) * cell_size
        cv2.circle(frame, (robot_x, robot_y), 3, [255, 255, 255], -1)

    def CreateLocalGridInfo(self):
        rows = int((kLocalGridForwardMax - kLocalGridForwardMin) / kLocalGridResolution) + 1
        cols = int((kLocalGridLateralMax - kLocalGridLateralMin) / kLocalGridResolution) + 1
        return {
            "grid": np.full((rows, cols), kGridUnknown, dtype=np.int8),
            "rows": rows,
            "cols": cols,
            "valid": False,
            "nearest_obstacle": None,
        }

    def LocalPointToGridIndex(self, forward, lateral, grid_info):
        row = int((forward - kLocalGridForwardMin) / kLocalGridResolution)
        col = int((lateral - kLocalGridLateralMin) / kLocalGridResolution)
        if row < 0 or row >= grid_info["rows"] or col < 0 or col >= grid_info["cols"]:
            return None
        return row, col

    def MarkLocalGridCell(self, grid_info, forward, lateral, value):
        index = self.LocalPointToGridIndex(forward, lateral, grid_info)
        if index is None:
            return False
        row, col = index
        grid_info["grid"][row, col] = value
        return True

    def MarkLocalGridRay(self, grid_info, forward, lateral):
        if forward <= 0.0:
            return

        free_forward = max(0.0, min(forward - kLocalGridResolution, kLocalGridForwardMax))
        steps = max(1, int(free_forward / kLocalGridResolution))
        for step in range(steps + 1):
            sample_forward = step * kLocalGridResolution
            if sample_forward > free_forward:
                break
            ratio = 0.0 if forward <= 0.0 else sample_forward / forward
            sample_lateral = lateral * ratio
            self.MarkLocalGridCell(grid_info, sample_forward, sample_lateral, kGridFree)

        if forward <= kLocalGridForwardMax:
            self.MarkLocalGridCell(grid_info, forward, lateral, kGridOccupied)
            if grid_info["nearest_obstacle"] is None or forward < grid_info["nearest_obstacle"]:
                grid_info["nearest_obstacle"] = forward

    def InflateLocalGrid(self, grid_info):
        grid = grid_info["grid"]
        occupied_indices = np.argwhere(grid == kGridOccupied)
        inflate_cells = int(math.ceil(kRobotSafetyRadius / kLocalGridResolution))
        for row, col in occupied_indices:
            row_min = max(0, row - inflate_cells)
            row_max = min(grid_info["rows"] - 1, row + inflate_cells)
            col_min = max(0, col - inflate_cells)
            col_max = min(grid_info["cols"] - 1, col + inflate_cells)
            for inflate_row in range(row_min, row_max + 1):
                for inflate_col in range(col_min, col_max + 1):
                    d_row = inflate_row - row
                    d_col = inflate_col - col
                    if math.sqrt(d_row * d_row + d_col * d_col) <= inflate_cells:
                        if grid[inflate_row, inflate_col] != kGridOccupied:
                            grid[inflate_row, inflate_col] = kGridInflated

    def BuildLocalGrid(self, depth_frame, intrinsics, image_width, image_height):
        grid_info = self.CreateLocalGridInfo()
        if depth_frame is None or intrinsics is None:
            return grid_info

        y_start = int(image_height * 0.35)
        y_end = int(image_height * 0.85)
        valid_count = 0
        for y in range(y_start, y_end, kDepthGridPixelStep):
            for x in range(0, image_width, kDepthGridPixelStep):
                depth = depth_frame.get_distance(x, y)
                if depth <= 0.0 or depth > kLocalGridForwardMax:
                    continue

                camera_x = (x - intrinsics.ppx) / intrinsics.fx * depth
                forward = depth
                lateral = -camera_x
                if lateral < kLocalGridLateralMin or lateral > kLocalGridLateralMax:
                    continue

                self.MarkLocalGridRay(grid_info, forward, lateral)
                valid_count += 1

        grid_info["valid"] = valid_count > 0
        if grid_info["valid"]:
            self.InflateLocalGrid(grid_info)
        return grid_info

    def GetLocalGridCellStatus(self, grid_info, forward, lateral):
        if grid_info is None or not grid_info["valid"]:
            return kGridUnknown
        index = self.LocalPointToGridIndex(forward, lateral, grid_info)
        if index is None:
            return kGridUnknown
        row, col = index
        return int(grid_info["grid"][row, col])

    def IsLocalTrajectorySafe(self, linear_velocity, radian_velocity, grid_info, sim_time=kPlannerSimTime):
        if linear_velocity <= 0.0:
            return True
        if grid_info is None or not grid_info["valid"]:
            return False

        forward = 0.0
        lateral = 0.0
        heading = 0.0
        steps = max(1, int(sim_time / kPlannerSimStep))
        for _ in range(steps):
            heading += radian_velocity * kPlannerSimStep
            forward += linear_velocity * math.cos(heading) * kPlannerSimStep
            lateral += linear_velocity * math.sin(heading) * kPlannerSimStep
            status = self.GetLocalGridCellStatus(grid_info, forward, lateral)
            if status != kGridFree:
                return False
        return True

    def GetOpenTurnDirection(self, grid_info):
        if grid_info is None or not grid_info["valid"]:
            return self.lost_search_direction

        left_score = 0
        right_score = 0
        row_min = int(0.2 / kLocalGridResolution)
        row_max = min(grid_info["rows"], int(1.2 / kLocalGridResolution))
        center_col = int((0.0 - kLocalGridLateralMin) / kLocalGridResolution)
        for row in range(row_min, row_max):
            for col in range(0, grid_info["cols"]):
                status = int(grid_info["grid"][row, col])
                if status != kGridFree:
                    continue
                if col < center_col:
                    right_score += 1
                else:
                    left_score += 1
        return 1 if left_score >= right_score else -1

    def ApplySafetyLimits(self, linear_velocity, radian_velocity, obstacle_info, grid_info):
        linear_velocity, radian_velocity, obstacle_info = self.ApplyObstacleAvoidance(
            linear_velocity, radian_velocity, obstacle_info)

        if linear_velocity > 0.0 and not self.IsLocalTrajectorySafe(linear_velocity, radian_velocity, grid_info, 0.8):
            obstacle_info["active"] = True
            obstacle_info["mode"] = "grid_block"
            return 0.0, self.GetOpenTurnDirection(grid_info) * kObstacleTurnVelocity, obstacle_info

        return linear_velocity, radian_velocity, obstacle_info

    def PlanLocalCommandToTarget(self, target_robot, grid_info, max_forward_velocity):
        if target_robot is None:
            return None, None

        target_forward, target_lateral = target_robot
        target_distance = math.sqrt(target_forward * target_forward + target_lateral * target_lateral)
        target_heading = math.atan2(target_lateral, target_forward)
        if target_distance < kPlannerTargetDeadband:
            return None, {
                "score": 0.0,
                "target_forward": target_forward,
                "target_lateral": target_lateral,
                "reached": True,
            }

        best_command = None
        best_score = None
        linear_candidates = [0.0, 0.10, 0.18, 0.26, max_forward_velocity]
        angular_candidates = [-0.80, -0.45, -0.20, 0.0, 0.20, 0.45, 0.80]
        for linear in linear_candidates:
            if linear > max_forward_velocity:
                continue
            for angular in angular_candidates:
                if abs(angular) > TransferConstants.kMaxRadianVelocity:
                    continue
                if not self.IsLocalTrajectorySafe(linear, angular, grid_info):
                    continue

                forward = 0.0
                lateral = 0.0
                heading = 0.0
                steps = max(1, int(kPlannerSimTime / kPlannerSimStep))
                for _ in range(steps):
                    heading += angular * kPlannerSimStep
                    forward += linear * math.cos(heading) * kPlannerSimStep
                    lateral += linear * math.sin(heading) * kPlannerSimStep

                end_dx = target_forward - forward
                end_dy = target_lateral - lateral
                end_distance = math.sqrt(end_dx * end_dx + end_dy * end_dy)
                end_heading_error = abs(math.atan2(end_dy, end_dx) - heading)
                end_heading_error = abs(math.atan2(math.sin(end_heading_error), math.cos(end_heading_error)))
                smooth_cost = abs(linear - self.last_linear_velocity)
                score = end_distance + 0.35 * end_heading_error + 0.20 * smooth_cost + 0.10 * abs(angular) - 0.15 * linear
                if best_score is None or score < best_score:
                    best_score = score
                    best_command = (linear, angular, linear)

        if best_command is None:
            return None, None

        return best_command, {
            "score": best_score,
            "target_forward": target_forward,
            "target_lateral": target_lateral,
        }

    def NormalizeAngle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def GetBranchSpecs(self):
        return [
            {"name": "left", "heading": math.pi / 2.0, "search_direction": 1},
            {"name": "front", "heading": 0.0, "search_direction": self.last_image_search_direction},
            {"name": "right", "heading": -math.pi / 2.0, "search_direction": -1},
        ]

    def GetPreferredBranchName(self):
        if self.last_image_search_direction > 0:
            return "left"
        if self.last_image_search_direction < 0:
            return "right"
        return "front"

    def ScoreBranchDirection(self, grid_info, branch_spec, predicted_target_robot=None):
        if grid_info is None or not grid_info["valid"]:
            return None

        heading = branch_spec["heading"]
        free_count = 0
        blocked_count = 0
        unknown_count = 0
        open_distance = 0.0
        for distance in np.arange(0.25, kBranchExploreMaxDistance + 0.05, kLocalGridResolution):
            distance_has_free = False
            distance_blocked = False
            for offset in [-0.20, 0.0, 0.20]:
                forward = math.cos(heading) * distance - math.sin(heading) * offset
                lateral = math.sin(heading) * distance + math.cos(heading) * offset
                status = self.GetLocalGridCellStatus(grid_info, forward, lateral)
                if status == kGridFree:
                    free_count += 1
                    distance_has_free = True
                elif status == kGridOccupied or status == kGridInflated:
                    blocked_count += 1
                    distance_blocked = True
                else:
                    unknown_count += 1
            if distance_has_free and not distance_blocked:
                open_distance = distance

        if free_count == 0 or blocked_count > free_count:
            return None

        predicted_bonus = 0.0
        if predicted_target_robot is not None:
            predicted_heading = math.atan2(predicted_target_robot[1], predicted_target_robot[0])
            heading_error = abs(self.NormalizeAngle(predicted_heading - heading))
            predicted_bonus = max(0.0, 1.0 - heading_error / math.pi)

        score = free_count + 3.0 * open_distance + 2.0 * predicted_bonus - 2.0 * blocked_count - 0.2 * unknown_count
        if score <= 0.0:
            return None

        return {
            "name": branch_spec["name"],
            "heading": heading,
            "search_direction": branch_spec["search_direction"],
            "score": score,
            "open_distance": open_distance,
            "free_count": free_count,
            "blocked_count": blocked_count,
            "unknown_count": unknown_count,
        }

    def BuildBranchQueue(self, grid_info, predicted_target_robot=None):
        scored_branches = []
        for branch_spec in self.GetBranchSpecs():
            branch_score = self.ScoreBranchDirection(grid_info, branch_spec, predicted_target_robot)
            if branch_score is not None:
                scored_branches.append(branch_score)

        preferred_name = self.GetPreferredBranchName()
        preferred = [branch for branch in scored_branches if branch["name"] == preferred_name]
        remaining = [branch for branch in scored_branches if branch["name"] != preferred_name]
        remaining.sort(key=lambda branch: branch["score"], reverse=True)
        return preferred + remaining

    def ResetBranchSearch(self):
        self.branch_queue = []
        self.current_branch = None
        self.branch_start_time = None
        self.branch_start_odom = None
        self.branch_phase = None
        self.branch_plan_initialized = False

    def StartNextBranch(self, odom_pose):
        if self.current_branch is not None:
            return True
        if len(self.branch_queue) == 0:
            return False
        self.current_branch = self.branch_queue.pop(0)
        self.branch_start_time = time.time()
        self.branch_start_odom = odom_pose
        self.branch_phase = "turn"
        self.lost_search_direction = self.current_branch["search_direction"]
        return True

    def GetBranchTraveledDistance(self, odom_pose):
        if odom_pose is None or self.branch_start_odom is None:
            return 0.0
        dx = odom_pose[0] - self.branch_start_odom[0]
        dy = odom_pose[1] - self.branch_start_odom[1]
        return math.sqrt(dx * dx + dy * dy)

    def FinishCurrentBranch(self):
        self.current_branch = None
        self.branch_start_time = None
        self.branch_start_odom = None
        self.branch_phase = None

    def CalculateBranchExploreCommand(self, odom_pose):
        if self.current_branch is None:
            return None

        now = time.time()
        elapsed = 0.0 if self.branch_start_time is None else now - self.branch_start_time
        if elapsed > kBranchExploreMaxDuration:
            self.FinishCurrentBranch()
            return None

        branch_heading = self.current_branch["heading"]
        if self.branch_phase == "turn":
            if odom_pose is not None and self.branch_start_odom is not None:
                target_yaw = self.branch_start_odom[2] + branch_heading
                heading_error = self.NormalizeAngle(target_yaw - odom_pose[2])
            else:
                heading_error = branch_heading

            if abs(heading_error) > kBranchTurnHeadingTolerance:
                return 0.0, max(-kBranchTurnVelocity, min(kBranchTurnVelocity, heading_error)), 0.0

            self.branch_phase = "forward"
            self.branch_start_time = now
            self.branch_start_odom = odom_pose

        traveled_distance = self.GetBranchTraveledDistance(odom_pose)
        if traveled_distance >= kBranchExploreMaxDistance:
            self.FinishCurrentBranch()
            return None

        return kBranchForwardVelocity, 0.0, kBranchForwardVelocity

    def GetLastSeenTargetRobot(self, odom_pose):
        return self.TransformOdomToRobot(self.last_seen_odom, odom_pose)

    def IsLastSeenReached(self, odom_pose):
        target_robot = self.GetLastSeenTargetRobot(odom_pose)
        if target_robot is None:
            return False
        target_forward, target_lateral = target_robot
        return math.sqrt(target_forward * target_forward + target_lateral * target_lateral) <= kLastSeenArriveDistance

    def CalculateBranchSearchCommand(self, grid_info, predicted_target_robot, odom_pose):
        if not self.branch_plan_initialized:
            self.branch_queue = self.BuildBranchQueue(grid_info, predicted_target_robot)
            self.branch_plan_initialized = True

        if self.current_branch is None:
            self.StartNextBranch(odom_pose)

        branch_command = self.CalculateBranchExploreCommand(odom_pose)
        if branch_command is None and self.current_branch is None and len(self.branch_queue) > 0:
            self.StartNextBranch(odom_pose)
            branch_command = self.CalculateBranchExploreCommand(odom_pose)

        return branch_command

    def ResetLostSearch(self):
        self.lost_search_start_time = None
        self.lost_search_last_switch_time = None
        self.lost_search_direction = self.last_image_search_direction

    def ResetTargetPrediction(self):
        self.target_history = []
        self.last_target_lost_time = None
        self.last_image_search_direction = 1
        self.predicted_target_odom = None
        self.prediction_confidence = 0.0
        self.last_seen_odom = None
        self.last_seen_time = None
        self.last_seen_arrived = False
        self.track_state = "lost_stop"
        self.last_reacquire_time = None
        self.last_reacquire_score = None
        self.ResetLostSearch()
        self.ResetBranchSearch()

    def CalculateLostSearchCommand(self, preferred_direction=None):
        now = time.time()
        if self.lost_search_start_time is None:
            self.lost_search_start_time = now
            self.lost_search_last_switch_time = now
            if preferred_direction is not None:
                self.lost_search_direction = preferred_direction
            else:
                self.lost_search_direction = self.last_image_search_direction

        if now - self.lost_search_start_time > kSearchMaxDuration:
            return None

        if now - self.lost_search_last_switch_time > kSearchSwitchInterval:
            self.lost_search_direction *= -1
            self.lost_search_last_switch_time = now

        return 0.0, self.lost_search_direction * kSearchRadianVelocity, 0.0

    def IsOdomValid(self, odom_pose, odom_status):
        if odom_pose is None:
            return False
        if odom_status is None:
            return True
        _, _, receive_count, age = odom_status
        if receive_count <= 0:
            return False
        if age is None:
            return True
        return age <= kOdomValidMaxAge

    def GetTargetVelocityOdom(self):
        valid_points = [item for item in self.target_history if item["person_odom"] is not None]
        if len(valid_points) < 2:
            return 0.0, 0.0

        last_point = valid_points[-1]
        for previous_point in reversed(valid_points[:-1]):
            dt = last_point["time"] - previous_point["time"]
            if dt >= 0.05:
                vx = (last_point["person_odom"][0] - previous_point["person_odom"][0]) / dt
                vy = (last_point["person_odom"][1] - previous_point["person_odom"][1]) / dt
                speed = math.sqrt(vx * vx + vy * vy)
                if speed > 1.5:
                    scale = 1.5 / speed
                    vx *= scale
                    vy *= scale
                return vx, vy
        return 0.0, 0.0

    def UpdateLastImageSearchDirection(self, center_x, image_width, now):
        direction = 1 if center_x < image_width * 0.5 else -1
        if len(self.target_history) > 0:
            previous = self.target_history[-1]
            dt = now - previous["time"]
            if dt > 0.01:
                image_velocity = (center_x - previous["center_x"]) / dt
                if abs(image_velocity) > image_width * 0.10:
                    direction = -1 if image_velocity > 0.0 else 1
        self.last_image_search_direction = direction

    def UpdateTargetObservation(self, box, center, person_distance, person_point, person_odom, image_width, image_height):
        now = time.time()
        center_x, center_y = center
        x1 = int(box.xyxy[0][0].item())
        y1 = int(box.xyxy[0][1].item())
        x2 = int(box.xyxy[0][2].item())
        y2 = int(box.xyxy[0][3].item())
        self.UpdateLastImageSearchDirection(center_x, image_width, now)

        observation = {
            "time": now,
            "center_x": center_x,
            "center_y": center_y,
            "center_norm_x": float(center_x) / max(1, image_width),
            "center_norm_y": float(center_y) / max(1, image_height),
            "box_width": max(1, x2 - x1),
            "box_height": max(1, y2 - y1),
            "distance": person_distance,
            "person_point": person_point,
            "person_odom": person_odom,
        }
        self.target_history.append(observation)
        if len(self.target_history) > kTargetHistoryMax:
            self.target_history.pop(0)

        if person_odom is not None and person_distance is not None:
            self.last_seen_odom = person_odom
            self.last_seen_time = now
            self.last_seen_arrived = False
            self.ResetBranchSearch()

        self.last_target_lost_time = None
        self.predicted_target_odom = None
        self.prediction_confidence = 1.0
        self.track_state = "track_visible"
        self.ResetLostSearch()
        self.ResetBranchSearch()

    def StartLostTargetTimer(self):
        if self.last_target_lost_time is None:
            self.last_target_lost_time = time.time()

    def PredictLostTargetOdom(self, odom_pose, odom_status):
        self.StartLostTargetTimer()
        now = time.time()
        lost_age = now - self.last_target_lost_time
        if lost_age > kPredictionMaxDuration:
            self.predicted_target_odom = None
            self.prediction_confidence = 0.0
            return None, 0.0, lost_age

        valid_points = [item for item in self.target_history if item["person_odom"] is not None]
        if len(valid_points) == 0 or not self.IsOdomValid(odom_pose, odom_status):
            self.predicted_target_odom = None
            confidence = max(0.0, 0.35 * (1.0 - lost_age / kOdomBlindMoveDuration))
            self.prediction_confidence = confidence
            return None, confidence, lost_age

        last_point = valid_points[-1]
        vx, vy = self.GetTargetVelocityOdom()
        horizon = max(kPredictionMinHorizon, min(kPredictionMaxHorizon, lost_age))
        predicted = (
            last_point["person_odom"][0] + vx * horizon,
            last_point["person_odom"][1] + vy * horizon,
        )
        confidence = max(0.0, 1.0 - lost_age / kPredictionMaxDuration)
        if abs(vx) < 0.01 and abs(vy) < 0.01:
            confidence *= 0.75

        self.predicted_target_odom = predicted
        self.prediction_confidence = confidence
        return predicted, confidence, lost_age

    def CalculateBlindImageSearchCommand(self, lost_age):
        if lost_age > kOdomBlindMoveDuration:
            return None
        angular_velocity = self.last_image_search_direction * kSearchRadianVelocity
        return kOdomBlindMaxForwardVelocity, angular_velocity, kOdomBlindMaxForwardVelocity

    def GetPathFallbackTargetRobot(self, odom_pose):
        path_target, _, _ = self.GetPathLookaheadTarget(odom_pose)
        return self.TransformOdomToRobot(path_target, odom_pose)

    def DeprojectPixelToPoint(self, intrinsics, pixel_x, pixel_y, depth):
        if intrinsics is None or depth is None or depth <= 0:
            return None

        camera_x = (pixel_x - intrinsics.ppx) / intrinsics.fx * depth
        camera_y = (pixel_y - intrinsics.ppy) / intrinsics.fy * depth
        camera_z = depth
        return camera_x, camera_y, camera_z

    def TransformPersonToOdom(self, person_forward, person_lateral, odom_pose):
        if person_forward is None or person_lateral is None or odom_pose is None:
            return None

        robot_x, robot_y, robot_yaw = odom_pose
        person_odom_x = robot_x + math.cos(robot_yaw) * person_forward - math.sin(robot_yaw) * person_lateral
        person_odom_y = robot_y + math.sin(robot_yaw) * person_forward + math.cos(robot_yaw) * person_lateral
        return person_odom_x, person_odom_y

    def TransformOdomToRobot(self, odom_point, odom_pose):
        if odom_point is None or odom_pose is None:
            return None

        robot_x, robot_y, robot_yaw = odom_pose
        dx = odom_point[0] - robot_x
        dy = odom_point[1] - robot_y
        target_forward = math.cos(robot_yaw) * dx + math.sin(robot_yaw) * dy
        target_lateral = -math.sin(robot_yaw) * dx + math.cos(robot_yaw) * dy
        return target_forward, target_lateral

    def GetPathLookaheadTarget(self, odom_pose):
        if odom_pose is None or len(self.person_path) == 0:
            return None, None, None

        robot_x, robot_y, _ = odom_pose
        closest_index = 0
        closest_distance_sq = None
        for index, point in enumerate(self.person_path):
            dx = point[0] - robot_x
            dy = point[1] - robot_y
            distance_sq = dx * dx + dy * dy
            if closest_distance_sq is None or distance_sq < closest_distance_sq:
                closest_distance_sq = distance_sq
                closest_index = index

        target_index = closest_index
        traveled_distance = 0.0
        last_point = self.person_path[closest_index]
        for index in range(closest_index + 1, len(self.person_path)):
            point = self.person_path[index]
            dx = point[0] - last_point[0]
            dy = point[1] - last_point[1]
            traveled_distance += math.sqrt(dx * dx + dy * dy)
            target_index = index
            if traveled_distance >= kPathLookaheadDistance:
                break
            last_point = point

        return self.person_path[target_index], closest_index, target_index

    def ClampRadianVelocity(self, heading_error):
        if heading_error is None or abs(heading_error) < kHeadingDeadband:
            return 0.0

        radian_velocity = heading_error * kAngularHeadingKp
        return max(
            -TransferConstants.kMaxRadianVelocity,
            min(TransferConstants.kMaxRadianVelocity, radian_velocity)
        )

    def CalculatePathCommand(self, odom_pose, max_forward_velocity):
        path_target, path_closest_index, path_target_index = self.GetPathLookaheadTarget(odom_pose)
        path_target_robot = self.TransformOdomToRobot(path_target, odom_pose)
        path_info = {
            "target": path_target,
            "closest_index": path_closest_index,
            "target_index": path_target_index,
            "target_robot": path_target_robot,
            "forward": None,
            "lateral": None,
            "distance": None,
            "heading_error": None,
            "target_linear_velocity": 0.0,
        }

        if path_target_robot is None:
            return None, path_info

        control_forward, control_lateral = path_target_robot
        control_distance = math.sqrt(control_forward * control_forward + control_lateral * control_lateral)
        heading_error = math.atan2(control_lateral, control_forward)
        path_info["forward"] = control_forward
        path_info["lateral"] = control_lateral
        path_info["distance"] = control_distance
        path_info["heading_error"] = heading_error

        if control_distance <= kPathTargetDeadband:
            return None, path_info

        radian_velocity = self.ClampRadianVelocity(heading_error)
        target_linear_velocity = 0.0
        if abs(heading_error) < kPathTurnInPlaceHeading and control_forward > kPathTargetDeadband:
            target_linear_velocity = control_distance * kLinearDistanceKp
            target_linear_velocity = min(max_forward_velocity, target_linear_velocity)
            turn_scale = max(0.2, 1.0 - min(1.0, abs(heading_error)))
            target_linear_velocity *= turn_scale

        path_info["target_linear_velocity"] = target_linear_velocity
        if target_linear_velocity <= 0.0:
            linear_velocity = 0.0
        else:
            linear_velocity = (
                kLinearVelocitySmooth * self.last_linear_velocity
                + (1.0 - kLinearVelocitySmooth) * target_linear_velocity
            )

        return (linear_velocity, radian_velocity, target_linear_velocity), path_info

    def ClearPersonPath(self):
        self.person_path = []
        self.ResetTargetPrediction()

    def AddPersonPathPoint(self, person_odom):
        if person_odom is None:
            return False

        if len(self.person_path) == 0:
            self.person_path.append(person_odom)
            return True

        last_x, last_y = self.person_path[-1]
        dx = person_odom[0] - last_x
        dy = person_odom[1] - last_y
        if math.sqrt(dx * dx + dy * dy) < kPathMinPointDistance:
            return False

        self.person_path.append(person_odom)
        if len(self.person_path) > kMaxPathPoints:
            self.person_path.pop(0)
        return True

    def DrawOdomPose(self, frame, odom_pose, y, odom_topic=None, odom_status=None):
        if odom_pose is None:
            receive_count = 0
            if odom_status is not None:
                _, _, receive_count, _ = odom_status
            cv2.putText(frame,"odom invalid recv {}".format(receive_count),(0,y), cv2.FONT_HERSHEY_PLAIN, 1.2, [0,0,255], 1)
            return

        robot_x, robot_y, robot_yaw = odom_pose
        topic_text = odom_topic if odom_topic is not None else "odom"
        qos_text = ""
        if odom_status is not None:
            _, qos_name, _, age = odom_status
            age_text = "--" if age is None else "{:.2f}s".format(age)
            qos_text = " qos {} age {}".format(qos_name, age_text)

        cv2.putText(frame,"odom x {:.2f} y {:.2f} yaw {:.2f}".format(robot_x, robot_y, robot_yaw),
                    (0,y), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
        cv2.putText(frame,"odom {}{}".format(topic_text, qos_text),
                    (0,y + 18), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
    
    def InputAndProcess(self, frame, key):
        if(self.GetIsTracking() == False):
            if key >= ord('0') and key <= ord('9'):  # 大键盘输入
                self.id_str += str((int(key - ord('0'))))
            elif key == ord('\b'):  # 退格
                self.id_str = self.id_str[:-1]
            elif key == 10 or key == 13 or key == 141:  # 回车
                if(self.id_str == ""):
                    self.SetTargetId(0)
                else:
                    self.SetTargetId(int(self.id_str))
                self.ClearPersonPath()
                self.id_str = ""

            if(self.GetTargetId() != kDefaultTrackId):
                self.SetIsTracking(True)

        else:
            if key == 10 or key == 13 or key == 141:  # 回车
                self.id_str = ""
                self.target_id = 0
                self.ClearPersonPath()
                self.SetIsTracking(False)
        return frame

    def TrackAndDraw(self, frame, box, depth_frame=None, color_intrinsics=None, odom_pose=None, odom_topic=None, odom_status=None, key=-1):
        shape = frame.shape
        frame = cv2.UMat(frame)
        local_grid_info = self.BuildLocalGrid(depth_frame, color_intrinsics, shape[1], shape[0])
        self.last_grid_info = local_grid_info
        planner_info = None
        planner_target_robot = None

        self.fps_counter.Count()
        frame = cv2.putText(frame, "fps {:.1f}".format(self.fps_counter.GetFps()), (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, [0, 128, 0], 1)
        self.InputAndProcess(frame, key)

        if(box != None):            
            x1 = int(box.xyxy[0][0].item())
            y1 = int(box.xyxy[0][1].item())
            x2 = int(box.xyxy[0][2].item())
            y2 = int(box.xyxy[0][3].item())
            center=((x1+x2)//2, (y1+y2)//2)
            person_distance, depth_region = self.GetBoxCenterDistance(
                depth_frame, x1, y1, x2, y2, shape[1], shape[0])
            person_point = self.DeprojectPixelToPoint(color_intrinsics, center[0], center[1], person_distance)
            cv2.circle(frame,center,2,[0,0,255],-1) # 画出选框中心点

            #cal target and velocity
            person_lateral = None
            person_forward = None
            person_odom = None
            person_heading_error = None
            path_target = None
            path_closest_index = None
            path_target_index = None
            path_target_robot = None
            control_mode = "person"
            control_forward = None
            control_lateral = None
            heading_error = None
            distance_error = None
            target_linear_velocity = 0.0
            linear_velocity = 0.0
            radian_velocity = 0.0
            if person_point is not None:
                person_lateral = -person_point[0]
                person_forward = person_point[2]
                person_odom = self.TransformPersonToOdom(person_forward, person_lateral, odom_pose)
                self.AddPersonPathPoint(person_odom)
                person_heading_error = math.atan2(person_lateral, person_forward)
                control_forward = person_forward
                control_lateral = person_lateral
            self.UpdateTargetObservation(box, center, person_distance, person_point, person_odom, shape[1], shape[0])

            path_command, path_info = self.CalculatePathCommand(odom_pose, kMaxForwardVelocity)
            path_target = path_info["target"]
            path_closest_index = path_info["closest_index"]
            path_target_index = path_info["target_index"]
            path_target_robot = path_info["target_robot"]
            if path_command is not None:
                control_mode = "path"
                linear_velocity, radian_velocity, target_linear_velocity = path_command
                control_forward = path_info["forward"]
                control_lateral = path_info["lateral"]
                heading_error = path_info["heading_error"]
                distance_error = path_info["distance"]
            elif person_point is not None and person_distance is not None:
                control_mode = "person"
                heading_error = person_heading_error
                radian_velocity = self.ClampRadianVelocity(heading_error)
                distance_error = person_distance - kTargetDistance
                if distance_error > kDistanceDeadband:
                    target_linear_velocity = distance_error * kLinearDistanceKp
                    target_linear_velocity = min(kMaxForwardVelocity, target_linear_velocity)
                    if heading_error is not None:
                        turn_scale = max(0.2, 1.0 - min(1.0, abs(heading_error)))
                        target_linear_velocity *= turn_scale
                    linear_velocity = (
                        kLinearVelocitySmooth * self.last_linear_velocity
                        + (1.0 - kLinearVelocitySmooth) * target_linear_velocity
                    )
            else:
                control_mode = "none"

            force_stop_by_distance = person_distance is not None and person_distance <= kTargetDistance
            if force_stop_by_distance:
                linear_velocity = 0.0
                target_linear_velocity = 0.0
            if distance_error is None and person_distance is not None:
                distance_error = person_distance - kTargetDistance

            obstacle_info = self.GetObstacleInfo(depth_frame, shape[1], shape[0])
            linear_velocity, radian_velocity, obstacle_info = self.ApplySafetyLimits(
                linear_velocity, radian_velocity, obstacle_info, local_grid_info)
            if obstacle_info["active"]:
                control_mode = "{}_avoid".format(control_mode)
            self.track_state = "track_visible"

            #draw
            label = '{}{:d}'.format("", self.GetTargetId())
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), [255,128,128], 2)
            if depth_region is not None:
                rx1, ry1, rx2, ry2 = depth_region
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), [0,255,0], 2)
            cv2.rectangle(frame,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), [255,128,128],-1)
            cv2.putText(frame,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)

            cv2.putText(frame,"ID {:} enter reset".format(self.GetTargetId()),(20,125), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            self.DrawOdomPose(frame, odom_pose, 50, odom_topic, odom_status)
            cv2.putText(frame,"v {:.2f} m/s".format(linear_velocity),(0,250), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            cv2.putText(frame,"w {:.2f} rad/s".format(radian_velocity),(0,270), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            self.DrawObstacleInfo(frame, obstacle_info, 410)
            self.DrawPlanningInfo(frame, local_grid_info, planner_info, planner_target_robot, shape[1], 430)
            if person_distance is not None:
                cv2.putText(frame,"depth {:.2f} target {:.2f}".format(person_distance, kTargetDistance),(0,290), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
                cv2.putText(frame,"mode {} err {:.2f} target_v {:.2f}".format(control_mode, distance_error, target_linear_velocity),(0,310), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
                if person_point is not None:
                    cv2.putText(frame,"rel x {:.2f} z {:.2f} head {:.2f}".format(person_lateral, person_forward, person_heading_error),(0,330), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
                    if person_odom is not None:
                        cv2.putText(frame,"person odom x {:.2f} y {:.2f}".format(person_odom[0], person_odom[1]),(0,350), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
                        cv2.putText(frame,"path n {:d} idx {}->{}".format(len(self.person_path), path_closest_index, path_target_index),(0,370), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
                if control_forward is not None and heading_error is not None:
                    cv2.putText(frame,"ctrl x {:.2f} z {:.2f} head {:.2f}".format(control_lateral, control_forward, heading_error),(0,390), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            else:
                cv2.putText(frame,"depth invalid",(0,290), cv2.FONT_HERSHEY_PLAIN, 1.2, [0,0,255], 1)

            #pub cmdvel
            if kUseRos1Transfer:
                self.ros1_transfer.SendCmdVel(linear_velocity, radian_velocity)
            else:
                self.ros2_transfer.SendCmdVel(linear_velocity, radian_velocity)
            self.last_linear_velocity = linear_velocity
            return frame
        else:
            path_command, path_info = self.CalculatePathCommand(odom_pose, kLostPathMaxForwardVelocity)
            control_mode = "lost_stop"
            linear_velocity = 0.0
            radian_velocity = 0.0
            target_linear_velocity = 0.0
            predicted_target_odom, prediction_confidence, lost_age = self.PredictLostTargetOdom(odom_pose, odom_status)
            planner_target_robot = None
            planner_command = None
            predicted_target_robot = None
            odom_valid = self.IsOdomValid(odom_pose, odom_status)
            if predicted_target_odom is not None and odom_valid:
                predicted_target_robot = self.TransformOdomToRobot(predicted_target_odom, odom_pose)

            if self.last_seen_odom is not None and odom_valid:
                planner_target_robot = self.GetLastSeenTargetRobot(odom_pose)
                if self.IsLastSeenReached(odom_pose):
                    self.last_seen_arrived = True
                elif not self.last_seen_arrived:
                    planner_command, planner_info = self.PlanLocalCommandToTarget(
                        planner_target_robot, local_grid_info, kLostPathMaxForwardVelocity)
                    control_mode = "lost_go_to_last_seen"
                    self.track_state = "lost_go_to_last_seen"
                    if planner_command is not None:
                        linear_velocity, radian_velocity, target_linear_velocity = planner_command

            if planner_command is None and self.last_seen_odom is not None and odom_valid:
                self.last_seen_arrived = True
                branch_command = self.CalculateBranchSearchCommand(
                    local_grid_info, predicted_target_robot, odom_pose)
                control_mode = "lost_branch_plan"
                self.track_state = "lost_branch_plan"
                if branch_command is not None:
                    linear_velocity, radian_velocity, target_linear_velocity = branch_command
                    control_mode = "lost_branch_explore"
                    self.track_state = "lost_branch_explore"

            if planner_command is None and self.last_seen_odom is None and predicted_target_robot is not None and prediction_confidence >= kPredictionMinConfidence:
                planner_target_robot = predicted_target_robot
                planner_command, planner_info = self.PlanLocalCommandToTarget(
                    planner_target_robot, local_grid_info, kLostPathMaxForwardVelocity)
                control_mode = "lost_predict"
                self.track_state = "lost_predict"
                if planner_command is not None:
                    linear_velocity, radian_velocity, target_linear_velocity = planner_command
                    control_mode = "lost_plan"
                    self.track_state = "lost_plan"

            if planner_command is None and self.last_seen_odom is None and odom_valid:
                fallback_target_robot = self.GetPathFallbackTargetRobot(odom_pose)
                if fallback_target_robot is not None:
                    planner_target_robot = fallback_target_robot
                    planner_command, planner_info = self.PlanLocalCommandToTarget(
                        planner_target_robot, local_grid_info, kLostPathMaxForwardVelocity)
                    if planner_command is not None:
                        linear_velocity, radian_velocity, target_linear_velocity = planner_command
                        control_mode = "lost_plan"
                        self.track_state = "lost_plan"

            if planner_command is None and not odom_valid:
                blind_command = self.CalculateBlindImageSearchCommand(lost_age)
                if blind_command is not None:
                    linear_velocity, radian_velocity, target_linear_velocity = blind_command
                    control_mode = "lost_predict"
                    self.track_state = "lost_predict"

            if planner_command is None and linear_velocity <= 0.0 and abs(radian_velocity) < 1e-6:
                search_command = self.CalculateLostSearchCommand(self.last_image_search_direction)
                if search_command is not None:
                    control_mode = "lost_scan"
                    self.track_state = "lost_scan"
                    linear_velocity, radian_velocity, target_linear_velocity = search_command
                else:
                    control_mode = "lost_stop"
                    self.track_state = "lost_stop"

            obstacle_info = self.GetObstacleInfo(depth_frame, shape[1], shape[0])
            linear_velocity, radian_velocity, obstacle_info = self.ApplySafetyLimits(
                linear_velocity, radian_velocity, obstacle_info, local_grid_info)
            if obstacle_info["active"]:
                control_mode = "{}_avoid".format(control_mode)

            if kUseRos1Transfer:
                self.ros1_transfer.SendCmdVel(linear_velocity, radian_velocity)
            else:
                self.ros2_transfer.SendCmdVel(linear_velocity, radian_velocity)
            self.last_linear_velocity = linear_velocity
            self.DrawOdomPose(frame, odom_pose, 50, odom_topic, odom_status)
            cv2.putText(frame,"lost ID {:}".format(self.GetTargetId()),(20,100), cv2.FONT_HERSHEY_PLAIN, 1.2, [0,0,255], 1)
            cv2.putText(frame,"enter reset",(20,120), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            cv2.putText(frame,"v {:.2f} m/s".format(linear_velocity),(0,250), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            cv2.putText(frame,"w {:.2f} rad/s".format(radian_velocity),(0,270), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            cv2.putText(frame,"mode {} target_v {:.2f}".format(control_mode, target_linear_velocity),(0,290), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            cv2.putText(frame,"path n {:d} idx {}->{}".format(len(self.person_path), path_info["closest_index"], path_info["target_index"]),(0,310), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            self.DrawObstacleInfo(frame, obstacle_info, 350)
            self.DrawPlanningInfo(frame, local_grid_info, planner_info, planner_target_robot, shape[1], 370)
            if path_info["forward"] is not None and path_info["heading_error"] is not None:
                cv2.putText(frame,"ctrl x {:.2f} z {:.2f} head {:.2f}".format(path_info["lateral"], path_info["forward"], path_info["heading_error"]),(0,330), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            return frame

    def NonTrackAndDraw(self, frame, odom_pose=None, odom_topic=None, odom_status=None, key=-1):
        frame = cv2.UMat(frame)
        self.fps_counter.Count()
        frame = cv2.putText(frame, "fps {:.1f}".format(self.fps_counter.GetFps()), (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, [0, 128, 0], 1)
        self.InputAndProcess(frame, key)
        if kUseRos1Transfer:
            self.ros1_transfer.SendCmdVel(0.0, 0.0)
        else:
            self.ros2_transfer.SendCmdVel(0.0, 0.0)
        self.last_linear_velocity = 0.0
        self.ResetTargetPrediction()
        self.DrawOdomPose(frame, odom_pose, 50, odom_topic, odom_status)
        cv2.putText(frame,"stop",(20,100), cv2.FONT_HERSHEY_PLAIN, 1.2, [0,0,255], 1)
        cv2.putText(frame,"input ID:",(20,120), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
        cv2.putText(frame,"v 0.00 m/s",(0,250), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
        cv2.putText(frame,"w 0.00 rad/s",(0,270), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
        return frame

    def Run(self, frame, depth_frame=None, color_intrinsics=None, odom_pose=None, odom_topic=None, odom_status=None, key=-1):
        results = self.yolo_wrapper.Track(frame)
        if(len(results)>0):
            if(self.GetIsTracking()):
                box = self.FindTarget(results[0].boxes)
                if box is None:
                    shape = frame.shape
                    box = self.ReacquireTarget(
                        results[0].boxes, depth_frame, color_intrinsics,
                        odom_pose, shape[1], shape[0])
                return self.TrackAndDraw(frame, box, depth_frame, color_intrinsics, odom_pose, odom_topic, odom_status, key)
            else:
                frame = results[0].plot()
                return self.NonTrackAndDraw(frame, odom_pose, odom_topic, odom_status, key)
        if(self.GetIsTracking()):
            return self.TrackAndDraw(frame, None, depth_frame, color_intrinsics, odom_pose, odom_topic, odom_status, key)
        return self.NonTrackAndDraw(frame, odom_pose, odom_topic, odom_status, key)
