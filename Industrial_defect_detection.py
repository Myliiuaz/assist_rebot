import math
import time
import numpy as np
from plugin_base import PluginBase

class IndustrialInspectionPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        # 工业专用属性
        self.prev_defect_pos = None
        self.defect_history = []
        self.inspection_start_time = 0
        self.last_inspection_time = None
        self.last_tracker_update_time = None
        
        # 将在on_load中初始化的属性
        self.tracker = None
        self.detector = None
        self.controller_utils = None
        self.console_util = None
        self.MAX_DT = None
        self.MIN_DT = None
        self.class_names = {}  # 类别ID到名称的映射

    def _ensure_detection_initialized(self):
        """确保 detection 相关的属性已初始化"""
        if self.detection is not None:
            if self.tracker is None:
                self.tracker = self.detection.tracker
            if self.detector is None:
                self.detector = self.detection.detector
            if self.controller_utils is None:
                self.controller_utils = self.detection.controller_utils
            if self.console_util is None:
                self.console_util = self.detection.console_util
            if self.MAX_DT is None:
                self.MAX_DT = self.detection.MAX_DT
            if self.MIN_DT is None:
                self.MIN_DT = self.detection.MIN_DT
            if not self.class_names and hasattr(self.detection, 'class_names'):
                self.class_names = self.detection.class_names

    def on_load(self):
        """插件加载时调用"""
        super().on_load()
        self._ensure_detection_initialized()
        # 初始化工业检测特定配置
        self.setup_industrial_config()
        
    def setup_industrial_config(self):
        """设置工业检测参数"""
        # 从配置加载或设置默认值
        self.min_defect_size = self.detection.config_manager.get('min_defect_size', 10)  # 最小缺陷尺寸(像素)
        self.max_defect_size = self.detection.config_manager.get('max_defect_size', 300)  # 最大缺陷尺寸
        self.critical_classes = self.detection.config_manager.get('critical_classes', [2, 3])  # 关键缺陷类别
        self.log_interval = self.detection.config_manager.get('log_interval', 5)  # 日志间隔(秒)
        self.last_log_time = time.time()
        
    def execute_main_logic(self, detection_boundary, min_confidence, reference_point):
        """执行工业检测逻辑"""
        self._ensure_detection_initialized()
        
        start_time = time.perf_counter()
        results = self.detector.detect(
            target_width=self.detection.screen_width,
            target_height=self.detection.screen_height
        )
        
        defects = []
        if not results:
            self.prev_defect_pos = None
            self.last_tracker_update_time = time.perf_counter()
            return None
        
        # 过滤和分类检测结果
        for obj in results:
            if obj.get('confidence') > min_confidence:
                width = obj['x2'] - obj['x1']
                height = obj['y2'] - obj['y1']
                size = max(width, height)
                
                # 尺寸过滤
                if self.min_defect_size <= size <= self.max_defect_size:
                    defect = {
                        "class_id": obj.get('class_id'),
                        "class_name": self.class_names.get(obj.get('class_id'), f"Class_{obj.get('class_id')}"),
                        "confidence": obj.get('confidence'),
                        "bbox": (obj['x1'], obj['y1'], obj['x2'], obj['y2']),
                        "center": ((obj['x1'] + obj['x2']) / 2, (obj['y1'] + obj['y2']) / 2),
                        "size": size,
                        "timestamp": time.time()
                    }
                    defects.append(defect)
        
        # 如果没有检测到缺陷
        if not defects:
            return None
        
        # 根据严重程度排序缺陷
        defects.sort(key=lambda x: (
            x['class_id'] in self.critical_classes,  # 关键缺陷优先
            x['size'],  # 尺寸大的优先
            x['confidence']  # 置信度高的优先
        ), reverse=True)
        
        # 更新缺陷历史记录
        self.update_defect_history(defects)
        
        # 获取主要缺陷
        primary_defect = defects[0]
        
        # 使用跟踪器预测缺陷位置（如果启用）
        if self.detection.prediction_config.get("enable", False):
            tracker_dt = start_time - (self.last_tracker_update_time or start_time)
            tracker_dt = max(min(tracker_dt, self.MAX_DT), self.MIN_DT)
            self.last_tracker_update_time = start_time
            self.tracker.update(primary_defect, dt=tracker_dt)
            predicted_pos = self.tracker.predict()
            if predicted_pos:
                primary_defect["bbox"] = predicted_pos
        
        return primary_defect
    
    def update_defect_history(self, defects):
        """更新缺陷历史记录"""
        current_time = time.time()
        # 保留最近30秒的记录
        self.defect_history = [
            d for d in self.defect_history 
            if current_time - d['timestamp'] <= 30
        ]
        self.defect_history.extend(defects)
        
        # 定期记录日志
        if current_time - self.last_log_time >= self.log_interval:
            self.log_inspection_results()
            self.last_log_time = current_time
    
    def log_inspection_results(self):
        """记录检测结果"""
        if not self.defect_history:
            return
        
        # 统计缺陷信息
        defect_stats = {}
        for defect in self.defect_history:
            cls = defect['class_name']
            defect_stats.setdefault(cls, {'count': 0, 'total_size': 0})
            defect_stats[cls]['count'] += 1
            defect_stats[cls]['total_size'] += defect['size']
        
        # 生成报告
        report = {"timestamp": time.time(), "defects": defect_stats}
        
        # 发送到控制台（实际应用中可能发送到数据库或监控系统）
        self.console_util.write_json(report)
        
        # 检查是否有严重缺陷
        critical_defects = [
            d for d in self.defect_history 
            if d['class_id'] in self.critical_classes
        ]
        
        if critical_defects:
            self.trigger_alert(critical_defects)
    
    def trigger_alert(self, critical_defects):
        """触发警报并控制硬件"""
        # 获取最严重的缺陷
        worst_defect = max(critical_defects, key=lambda x: x['size'])
        
        # 在控制台显示警报
        alert_msg = (
            f"CRITICAL DEFECT ALERT! "
            f"Type: {worst_defect['class_name']}, "
            f"Size: {worst_defect['size']}px, "
            f"Confidence: {worst_defect['confidence']*100:.1f}%"
        )
        self.console_util.write(alert_msg)
        
        # 控制工业设备 - 示例：停止传送带并标记位置
        self.control_industrial_equipment(worst_defect)
    
    def control_industrial_equipment(self, defect):
        """控制工业设备响应缺陷"""
        # 1. 停止传送带
        self.controller_utils.send_industrial_command("CONVEYOR_STOP")
        
        # 2. 计算机械臂坐标
        arm_x, arm_y = self.calculate_arm_coordinates(defect['center'])
        
        # 3. 移动机械臂到缺陷位置
        self.controller_utils.send_industrial_command(f"ARM_MOVE_TO {arm_x} {arm_y}")
        
        # 4. 标记缺陷
        self.controller_utils.send_industrial_command("MARK_DEFECT")
        
        # 5. 记录到数据库
        defect_record = {
            "type": defect['class_name'],
            "position": defect['center'],
            "size": defect['size'],
            "confidence": defect['confidence'],
            "timestamp": time.time()
        }
        self.console_util.write_json({"defect_record": defect_record})
    
    def calculate_arm_coordinates(self, image_position):
        """将图像位置转换为机械臂坐标"""
        # 在实际应用中，这里会有复杂的坐标转换逻辑
        # 简化的线性映射示例
        x, y = image_position
        arm_x = (x / self.detection.screen_width) * self.detection.arm_range_x
        arm_y = (y / self.detection.screen_height) * self.detection.arm_range_y
        return arm_x, arm_y
    
    def post_execution_logic(self, ai_config_manager):
        """后处理逻辑 - 在工业场景中用于处理控制命令"""
        self._ensure_detection_initialized()
        
        # 获取检测到的主要缺陷
        primary_defect = self.detection.get_detect_result()
        
        if primary_defect:
            # 实时显示缺陷信息（可选）
            self.display_defect_info(primary_defect)
            
            # 检查是否需要立即处理
            if primary_defect['class_id'] in self.critical_classes:
                self.handle_critical_defect(primary_defect)
    
    def display_defect_info(self, defect):
        """在控制台显示缺陷信息"""
        info = (
            f"Defect detected: {defect['class_name']}\n"
            f"Position: ({defect['center'][0]:.1f}, {defect['center'][1]:.1f})\n"
            f"Size: {defect['size']}px, Confidence: {defect['confidence']*100:.1f}%"
        )
        self.console_util.write(info)
    
    def handle_critical_defect(self, defect):
        """处理关键缺陷"""
        # 立即停止生产线（如果未在之前的逻辑中处理）
        #self.controller_utils.send_industrial_command("EMERGENCY_STOP")
        
        # 通知操作员
        #self.notify_operator(defect)
    
    def notify_operator(self, defect):
        """通知操作员"""
        message = (
            "紧急通知：检测到关键缺陷！\n"
            f"类型: {defect['class_name']}\n"
            f"位置: X={defect['center'][0]:.0f}, Y={defect['center'][1]:.0f}\n"
            f"尺寸: {defect['size']}像素"
        )
        
        # 通过多种方式通知
        self.console_util.write(message)  # 控制台显示
        #self.controller_utils.send_industrial_command(f"ALARM {message}")  # 现场警报
        #self.send_network_notification(message)  # 网络通知
    
    def send_network_notification(self, message):
        """发送网络通知"""
        # 实际实现中会连接到工厂的MES/SCADA系统
        notification = {
            "type": "defect_alert",
            "message": message,
            "timestamp": time.time(),
            "severity": "critical"
        }
        self.console_util.write_json(notification)
