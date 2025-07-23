import math
import time
import numpy as np

from plugin_base import PluginBase


class SimpleAimingPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.fire_state = False
        self.prev_target_pos = None
        self.smoothed_velocity = (0, 0)
        self.is_aiming_active = False
        self.aim_activation_time = 0
        self.last_pid_update_time = None
        self.last_tracker_update_time = None
        self.last_target_time = None
        self.prev_target_pos = None
        self.smoothed_velocity = (0.0, 0.0)
        self.is_aiming_active = True
        self.aim_activation_time = time.perf_counter()
        self.last_pid_update_time = time.perf_counter()
        self.last_tracker_update_time = None
        self.last_target_time = None
        self.last_aim_start = time.time()
        
        # 不要在 __init__ 中访问 self.detection 的属性
        # 这些将在 on_load 或首次使用时初始化
        self.tracker = None
        self.out_controller = None
        self.controller_utils = None
        self.MAX_DT = None
        self.MIN_DT = None

    def _ensure_detection_initialized(self):
        """确保 detection 相关的属性已初始化"""
        if self.detection is not None and self.tracker is None:
            self.tracker = self.detection.tracker
            self.out_controller = self.detection.out_controller
            self.controller_utils = self.detection.controller_utils
            self.detector = self.detection.detector
            self.console_util = self.detection.console_util
            self.MAX_DT = self.detection.MAX_DT
            self.MIN_DT = self.detection.MIN_DT

    def on_load(self):
        """插件加载时调用 - 在这里初始化依赖于 detection 的属性"""
        super().on_load()
        self._ensure_detection_initialized()

    def execute_main_logic(self, detection_boundary, min_confidence, crosshair_pos):
        # 确保初始化
        self._ensure_detection_initialized()
        
        start_time = time.perf_counter()
        results = self.detector.detect(target_width=self.detection.screen_width,target_height=self.detection.screen_height)
        enemies = []
        if not results:
            # 目标丢失时重置速度
            self.smoothed_velocity = (0.0, 0.0)
            self.prev_target_pos = None
            self.last_target_time = time.perf_counter()
            # self.last_pid_update_time = time.perf_counter()
            self.last_tracker_update_time = time.perf_counter()
            return None
        
        for obj in results:
            if obj.get('class_id') in self.detection.select_class:
                enemies.append({
                    "confidence": obj.get('confidence'),
                    "bbox": (obj.get('x1'), obj.get('y1'), obj.get('x2'), obj.get('y2')),
                    "cls": obj.get('class_id'),
                    "center": ((obj.get('x1')+obj.get('x2'))/2, (obj.get('y1')+obj.get('y2'))/2)
                })
        evaluated = self.detection.threat_evaluator.evaluate(enemies, self.detection.priority_class)
        target = evaluated[0]
        # 目标写入到输出面板 调试时使用，暂时不用注释掉
        # self.console_util.write_json(target, ensure_ascii=False)

        if self.is_aiming_active:
            current_center = target["position"]
            self.calculate_smoothed_velocity(current_center=current_center, start_time=start_time)

        else:
            self.smoothed_velocity = (0.0, 0.0)
            self.prev_target_pos = None

        if self.detection.prediction_config.get("enable"):
            tracker_dt = start_time - self.last_tracker_update_time
            self.last_tracker_update_time = start_time
            tracker_dt = max(min(tracker_dt, self.MAX_DT), self.MIN_DT)
            self.tracker.update(target, dt=tracker_dt)
            tracker_pos = self.tracker.predict()
            if tracker_pos:
                target["bbox"] = tracker_pos

        if self.detection.target_pos_radio.get("target_pos_radio", 0):
            aim_x, aim_y = self.detection.aim_generator.update(target["bbox"], target["cls"],
                                                target["confidence"])
        else:
            aim_x, aim_y = self.calculate_target_point(target["bbox"], self.detection.target_pos_radio.get("x", 1), self.detection.target_pos_radio.get("y", 1))
        if self.out_controller:
            actual_dt = start_time - self.last_pid_update_time
            actual_dt = max(min(actual_dt, self.MAX_DT), self.MIN_DT)
            self.last_pid_update_time = start_time
            distance_factor = 1.0
            if target and self.detection.adaptiveScale_config.get('enable', 0) == 1:
                bbox_area = (target.get("bbox")[1]-target.get("bbox")[0])*(target.get("bbox")[3]-target.get("bbox")[2])
                aspect_ratio = (target.get("bbox")[1]-target.get("bbox")[0])/(target.get("bbox")[3]-target.get("bbox")[2])
                distance_factor = self.detection.fps_auto_scale.calculate_scale(bbox_area, aspect_ratio, target.get("confidence", 0.4))
            aim_x, aim_y = self.out_controller.update(
                target_pos=(aim_x, aim_y),
                crosshair_pos=self.detection.screen_center,
                dt=actual_dt,
                fire_state=self.fire_state,
                distance_factor=distance_factor,
                mouse_status=self.detection.mouse_status,
                target_velocity=self.smoothed_velocity if self.is_aiming_active else (0,0)
            )
        return (aim_x, aim_y)

    async def post_execution_logic(self, ai_config_manager):
        self._ensure_detection_initialized()
        
        button_lr, btn_map = self.detection.controller_utils.get_active_btoon(self.detection.bind_btn)
        self.detection.mouse_status = btn_map
        if button_lr:
            is_left = button_lr == 1
            self.detection.fire_state = is_left
            self.on_mouse_down()

            ranked_targets = self.detection.get_detect_result()
            if ranked_targets:
                self.detection.controller_utils.execute_mouse_movement(ranked_targets, self.detection)
        else:
            self.on_mouse_up()

    def on_mouse_down(self):
        """鼠标按下事件处理"""
        if not self.is_aiming_active:
            self.prev_target_pos = None
            self.smoothed_velocity = (0.0, 0.0)
            self.is_aiming_active = True
            self.aim_activation_time = time.perf_counter()
            self.last_pid_update_time = time.perf_counter()
            self.last_tracker_update_time = None
            self.last_target_time = None
            self.tracker.reset()
            self.out_controller.reset()
            self.last_aim_start = time.time()

    def on_mouse_up(self):
        """鼠标释放事件处理"""
        self.is_aiming_active = False

    def calculate_distance_factor(self, bbox, cls, current_speed=0):
        current_width, current_height = self.detection.screen_width, self.detection.screen_height
        screen_area = current_width * current_height

        # 目标面积计算
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        # 动态计算区域阈值（无需修改）
        f_cfg = self.detection.far_pid_radio_cfg.get(cls, self.detection.far_pid_radio_cfg[0])
        n_cfg = self.detection.near_pid_radio_cfg.get(cls, self.detection.near_pid_radio_cfg[0])

        f_min_area = int(screen_area * f_cfg["min_radio"])
        f_max_area = int(screen_area * f_cfg["max_radio"])
        n_min_area = int(screen_area * n_cfg["min_radio"])
        n_max_area = int(screen_area * n_cfg["max_radio"])

        # 动态调整衰减系数
        resolution_scale = current_width / 1920
        near_coeff = self.detection.near_attenuation_coefficient * (1080 / current_height)
        far_coeff = self.detection.far_attenuation_coefficient * (1080 / current_height)

        # 计算 distance_factor
        distance_factor = 1.0
        if n_min_area < area < n_max_area:
            normalized = (n_max_area - area) / (n_max_area - n_min_area)
            normalized = max(0.0, min(normalized, 1.0))
            distance_factor = math.pow(normalized, near_coeff)
        elif f_min_area < area < f_max_area:
            normalized = (area - f_min_area) / (f_max_area - f_min_area)
            normalized = max(0.0, min(normalized, 1.0))
            distance_factor = math.pow(normalized, far_coeff)

        # 分辨率补偿 + 动态钳位
        distance_factor *= resolution_scale
        clamp_max = 3.0 * math.sqrt(resolution_scale)
        return max(0.1, min(distance_factor, clamp_max))

    def calculate_smoothed_velocity(self, current_center, start_time):
        x_smoothed_velocity = 0.0
        y_smoothed_velocity = 0.0
        # 计算原始速度（像素/秒）
        if self.prev_target_pos is not None:
            dt_speed = start_time - self.last_target_time
            dt_speed = max(min(dt_speed, self.MAX_DT), self.MIN_DT)
            dx = current_center[0] - self.prev_target_pos[0]
            alpha = self.detection.x_before_controller.get("alpha", 0.3)
            if abs(dx) < self.detection.x_before_controller.get("min_movement", 3):
                x_smoothed_velocity = self.smoothed_velocity[0] * alpha
            else:
                raw_vx = dx / dt_speed if dt_speed > 1e-5 else 0.0
                x_smoothed_velocity = alpha * raw_vx + (1 - alpha) * self.smoothed_velocity[0]
                x_smoothed_velocity = np.clip(x_smoothed_velocity,
                                              -self.detection.x_before_controller.get("max_plausible_speed", 1500),
                                              self.detection.x_before_controller.get("max_plausible_speed", 1500))

                elapsed = start_time - self.aim_activation_time
                ramp_factor = min(elapsed / self.detection.x_before_controller.get("activation_ramp_time", 0.1), 1.0)
                x_smoothed_velocity = x_smoothed_velocity * ramp_factor
                self.prev_target_pos[0] = current_center[0]
                self.last_target_time = start_time
            dy = current_center[1] - self.prev_target_pos[1]
            alpha = self.detection.y_before_controller.get("alpha", 0.3)
            if abs(dy) < self.detection.y_before_controller.get("min_movement", 3):
                y_smoothed_velocity = self.smoothed_velocity[1] * alpha
            else:
                raw_vy = dy / dt_speed if dt_speed > 1e-5 else 0.0
                y_smoothed_velocity = alpha * raw_vy + (1 - alpha) * self.smoothed_velocity[1]
                y_smoothed_velocity = np.clip(y_smoothed_velocity,
                                              -self.detection.y_before_controller.get("max_plausible_speed", 1500),
                                              self.detection.y_before_controller.get("max_plausible_speed", 1500))

                # 激活渐变
                elapsed = start_time - self.aim_activation_time
                ramp_factor = min(elapsed / self.detection.y_before_controller.get("activation_ramp_time", 0.1), 1.0)
                y_smoothed_velocity = y_smoothed_velocity * ramp_factor
                self.prev_target_pos[1] = current_center[1]
                self.last_target_time = start_time

        self.smoothed_velocity = (x_smoothed_velocity, y_smoothed_velocity)

    def calculate_target_point(self, boxes, x, y):
        x1, y1, x2, y2 = boxes[0], boxes[1], boxes[2], boxes[3]

        # 计算x坐标
        x_center = (x1 + x2) / 2
        if x <= 1:
            x_coord = x1 + (x_center - x1) * x  # x在[0,1]区间时从左到中心
        else:
            x_coord = x_center + (x2 - x_center) * (x - 1)  # x在(1,2]区间时从中心到右

        # 计算y坐标
        y_center = (y1 + y2) / 2
        if y <= 1:
            y_coord = y2 - (y2 - y_center) * y  # y在[0,1]区间时从下到中心
        else:
            y_coord = y_center - (y_center - y1) * (y - 1)  # y在(1,2]区间时从中心到上

        return (int(x_coord), int(y_coord))
