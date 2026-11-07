"""ROS adapter for the bounded real-vehicle Ackermann DWA and supervisor.

/localization and obstacle coordinates are geometric centres in map. Core and
/rda/traj use rear axle poses internally; no localization publication/TF changes.
A spawned process evaluates steering-target candidates. The main 10 Hz timer
always supervises freshness and mode sessions, even while a plan is pending.
/rda/plan carries an atomic reference, session, mode, target and validity reason;
legacy topics remain available for diagnostics. Only tracker publishes /car_cmd.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

# 算法包 rda_dwa 不在系统 site-packages 时，允许通过 RDA_DWA_SRC 指定源码目录
# （launch 已设 PYTHONPATH；此处覆盖 ros2 run / 直接 python 启动的场景）
_rda_src = os.environ.get('RDA_DWA_SRC', '')
if _rda_src and os.path.isdir(_rda_src) and _rda_src not in sys.path:
    sys.path.insert(0, _rda_src)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32MultiArray, String, Float32, Int32
from sensor_msgs.msg import LaserScan, NavSatFix
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
try:
    from control_msgs.msg import CarState   # 底盘轮速反馈（/car_state）
except Exception:
    CarState = None
try:
    from control_msgs.msg import CarCtr     # tracker 实际下发指令（/car_cmd.speed，V-streak 计数源；
                                            # 与 traj_tracker_node/car_ctr.cpp/diag_recorder 同为 control_msgs，
                                            # 原误写 car_ctr_msgs（包不存在）致 CarCtr=None、跳过 /car_cmd 订阅）
except Exception:
    CarCtr = None

from rda_dwa.module_config import module_config
from rda_dwa.enhanced_dwa import EnhancedDWA
from rda_dwa.realtime_ackermann import RealtimeAckermannPlanner
from .realtime_vehicle_core import RealtimeVehicleBridgeCore
from .async_planner import AsyncPlanner
from rda_dwa.four_wheel_steer_model import FourWheelSteerParams
from rda_dwa.ackermann_model import AckermannParams, AckermannModel
from rda_dwa.predictor import ObstaclePredictor

from .vehicle_bridge import (VehicleBridgeCore, BridgeConfig, compute_spin_target,
                             wrap_angle)
from .obstacle_fusion import Clusterer, cluster_bounding_box, infer_type
from .frame_utils import quat_to_yaw, is_body_frame, body_pose_to_base_link
from .dqn_infer import DQNPolicy
from .telemetry_recording import safe_data
from .localization_input import LocalizationInput
# 2026-11-06 重构：/rda/traj 改为"前方 2.5m 路基"（里程域积分，见 vehicle_bridge._build_road_base），
# 不再使用 predict_reference_traj / traj_point_count（定时域预测）与 traj_cache（时序缓存，已下移跟踪层）。

# 自定义消息（rda_dwa_real_msgs 包：msg/Obstacle.msg, ObstacleArray.msg）
try:
    from rda_dwa_real_msgs.msg import ObstacleArray
except Exception:  # 未 build 前允许 import（单测/语法检查）
    ObstacleArray = None


class RealVehicleBridgeNode(Node):
    """ROS2 实车适配节点。"""

    def __init__(self):
        super().__init__('rda_dwa_real_bridge')
        self._localization_input = LocalizationInput(self)

        # ---------------- 参数 ----------------
        self.declare_parameter('algorithm_frame', 'map')
        self.declare_parameter('control_rate', 10.0)
        self.declare_parameter('body_frames', ['base_link', 'base_footprint',
                                               'laser', 'lidar', 'sensor'])
        self.declare_parameter('sensor_frames', ['laser', 'lidar', 'sensor',
                                                 'laser_frame', 'lidar_frame'])
        self.declare_parameter('sensor_x', 0.0)
        self.declare_parameter('sensor_y', 0.0)
        self.declare_parameter('sensor_yaw', 0.0)
        # Localization is already the confirmed body centre; rear conversion is inside core.
        self.declare_parameter('body_to_base_link_x', 0.0)
        self.declare_parameter('localization_timeout', 1.0)
        self.declare_parameter('obstacle_timeout', 2.0)
        self.declare_parameter('wheel_feedback_timeout', 0.3)  # 底盘轮速反馈新鲜度（s）
        self.declare_parameter('car_state_topic', '/car_state')        # 底盘四轮轮速反馈
        self.declare_parameter('bridge_state_topic', '/car_bridge_state')  # 控制桥执行状态
        self.declare_parameter('emergency_stop', True)
        self.declare_parameter('enable_trap', True)
        # ---- 模式仲裁参数（2026-11-04：SPIN/trap 判定上收 planner 侧；
        #   与 vehicle_bridge.BridgeConfig 默认一致，yaml rda_dwa_real_bridge 段可覆盖）----
        self.declare_parameter('v_streak_threshold', 0.05)
        self.declare_parameter('v_streak_count', 25)
        self.declare_parameter('ack_guard_time', 2.0)
        self.declare_parameter('spin_reset_progress', 0.5)
        self.declare_parameter('trap_exit_dist', 0.3)
        self.declare_parameter('trap_exit_goal_ang_deg', 20.0)
        self.declare_parameter('trap_exit_sector_deg', 60.0)
        self.declare_parameter('trap_exit_sector_range', 2.5)
        self.declare_parameter('trap_exit_sector_max_obs', 3)
        self.declare_parameter('trap_max_duration', 16.0)   # trap 兜底最大持续 s（替代 120 步=12s）
        self.declare_parameter('obstacle_max_speed', 1.5)
        self.declare_parameter('cluster_gate', 0.6)
        self.declare_parameter('cluster_min_points', 2)
        self.declare_parameter('cluster_max_radius', 30.0)
        self.declare_parameter('track_gate', 1.0)
        self.declare_parameter('track_lost_ttl', 5)
        # 算法车辆参数（实车几何/运动学）
        self.declare_parameter('vehicle_length', 1.2)
        self.declare_parameter('vehicle_width', 0.8)
        self.declare_parameter('v_max', 0.6)
        self.declare_parameter('v_min', 0.0)
        self.declare_parameter('omega_max', 1.0)
        self.declare_parameter('omega_spin_max', 0.8)
        self.declare_parameter('a_max', 1.0)
        self.declare_parameter('dec_max', 1.5)
        self.declare_parameter('omega_acc', 1.0)
        self.declare_parameter('delta_max_deg', 60.0)
        self.declare_parameter('goal_tolerance', 0.3)
        # ---- A 方案：阿克曼运动学（真实车，2026-11-01 用户确认参数） ----
        # kinematics_mode: 'ackermann'（默认）→ 规划用 rectangle（Ackermann 非完整约束，
        #   _eff_omega 曲率钳位 + 多点掉头兜底，禁用原地自转）；'4wis' → 回退旧 4WIS 行为。
        self.declare_parameter('kinematics_mode', 'ackermann')
        self.declare_parameter('ack_wheelbase', 0.69)         # 轴距 m（实车确认）
        self.declare_parameter('ack_length', 1.2)             # 车长 m（撞检矩形；按 lidar 自剔除盒 1.2×0.8，实测可改）
        self.declare_parameter('ack_width', 0.8)              # 车宽 m
        self.declare_parameter('ack_delta_max_deg', 30.0)     # 前轮最大转角 deg（实车确认）
        self.declare_parameter('ack_delta_dot_max_deg', 30.0) # 转向角速度上限 deg/s
        self.declare_parameter('ack_v_max', 0.6)              # 物理线速度上限 m/s
        self.declare_parameter('ack_v_min', 0.0)             # 倒车上限 m/s（规划允许倒车）
        self.declare_parameter('ack_a_max', 0.7)              # 加速度 m/s²
        self.declare_parameter('ack_dec_max', 1.2)            # 减速度 m/s²
        # ---- 算法时域（实车固定时域；用户指定 traj_predict_mode='fixed'）----
        self.declare_parameter('traj_predict_mode', 'fixed')
        self.declare_parameter('sim_time', 3.0)   # ★2026-11-04 与 params.py:66/yaml 对齐 1.5→3.0
        # /rda/traj 发布用积分参数（2026-11-03 第二轮：从硬编码 0.05/0.05 参数化）
        self.declare_parameter('traj_dt', 0.05)       # 积分步长 s（sim_time/dt=60 点）
        self.declare_parameter('traj_v_eps', 0.05)    # |v| 低于此视为静止 → 轨迹长度≈0
        # ---- 方案B 时序缓存已废弃（2026-11-06）：融合/平滑下移到跟踪层（里程域接续融合 +
        # B 样条）。下面三个键不再读取，仅为兼容旧 yaml 保留声明。----
        self.declare_parameter('traj_cache_enable', True)
        self.declare_parameter('traj_cache_past_margin', 0.2)
        self.declare_parameter('traj_cache_future_margin', 0.2)
        self.declare_parameter('smooth_weight', 0.3)    # → planner.params.alpha_smooth（轨迹平滑权重）
        self.declare_parameter('sticky_eps', 0.05)    # → planner.params.sticky_eps（指令黏滞死区 m/s）
        # ---- APF 自转目标角参数（/rda/spin_target 用；v15_latesc 构建不覆盖 params.py 默认 1.0/0.6/1.5）----
        self.declare_parameter('apf_k_att', 1.0)
        self.declare_parameter('apf_k_rep', 0.6)
        self.declare_parameter('apf_d0', 1.5)
        self.declare_parameter('spin_request_timeout', 15.0)  # P2-修复④：SPIN_REQUEST 未见 tracker 执行的最长等待 s
        # ---- ★2026-11-06 用户口述逻辑参数（死区保持 / 5帧判据 / trap 两套退出 / 路基）----
        self.declare_parameter('ack_hold_enable', True)      # 输出 1.5×分辨率死区保持开关
        self.declare_parameter('ack_hold_v_ratio', 1.5)      # v 死区倍数（×v_resolution）
        self.declare_parameter('ack_hold_w_ratio', 1.5)      # ω 死区倍数（×omega_resolution）
        self.declare_parameter('ack_bad_count', 5)           # 连续 N 帧不满足阿克曼运动学 → 切自转
        self.declare_parameter('ack_bad_ratio', 0.02)        # 曲率超限判定余量
        self.declare_parameter('spin_count_for_trap', 2)     # 自转激活次数 ≥此 才考虑 trap
        self.declare_parameter('spin_gap_eps', 0.5)          # 最近两次自转点间距 (<此 → trap)
        self.declare_parameter('stall_obs_dist', 2.5)        # 卡住判据：最近障碍距离 m
        self.declare_parameter('stall_escape_time', 0.5)     # 卡住持续 s → 计一帧不可行
        self.declare_parameter('trap_exit_far', 1.5)         # 激活距离 >此 → 用绝对 0.3m 退出
        self.declare_parameter('trap_exit_ratio', 0.25)      # 否则用激活距离的 1/4
        self.declare_parameter('trap_exit_ang_deg', 60.0)    # 条件2a：车头与子目标连线夹角 °
        self.declare_parameter('trap_exit_block_max', 0)     # 条件2b：转向楔形内障碍数上限
        self.declare_parameter('road_base_len', 2.5)         # 路基长度 m
        self.declare_parameter('road_base_ds', 0.1)          # 路基弧长采样步长 m
        # APF 合力方向退化保护：合力≈车头方向(斥引力共线/障碍出影响域)时改用自由空间方向
        self.declare_parameter('spin_target_free_fallback', True)
        # ---- DQN 模型版（可选）：model_path 指向 export_dqn_npz.py 导出的 .npz ----
        self.declare_parameter('model_path', '')
        self.declare_parameter('model_input_dim', 0)   # 0=从权重自动读取（20/21/25）
        self.declare_parameter('model_state_dt', 0.1)  # make_state 步长（与训练一致）
        # ---- GPS 定位（无雷达/相机时：外部感知话题 + GPS 定位）----
        self.declare_parameter('gps_topic', '/fix')          # sensor_msgs/NavSatFix
        self.declare_parameter('gps_yaw_topic', '')           # 可选：std_msgs/Float32 航向(rad)
        self.declare_parameter('gps_yaw_use_diff', True)      # 无航向话题时用 GPS 位移差分估航向

        for key,default in {'planning_budget_s':.08,'safety_margin':.12,'heading_weight':.3,'steer_weight':.12,'steer_change_weight':.08,'guide_extra':.08,'command_acc':.7,'command_steer_rate_deg':30.,'near_goal_spin_distance':4.,'near_goal_spin_angle_deg':65.,'blocked_spin_time':1.2,'spin_speed':.6,'cruise_velocity':1.,'lateral_acc_max':.6,'goal_replay_epsilon':.05,'motion_stall_time':3.,'recovery_subgoal_distance':1.8,'recovery_max_retries':2,'goal_max_spin_requests':4}.items():
            self.declare_parameter(key,default)
        self.declare_parameter('async_planning', True)
        self.declare_parameter('velocity_horizon_targets',True)
        cfg = BridgeConfig()
        cfg.algorithm_frame = self.get_parameter('algorithm_frame').value
        cfg.body_frames = list(self.get_parameter('body_frames').value)
        cfg.sensor_frames = list(self.get_parameter('sensor_frames').value)
        cfg.sensor_x = float(self.get_parameter('sensor_x').value)
        cfg.sensor_y = float(self.get_parameter('sensor_y').value)
        cfg.sensor_yaw = float(self.get_parameter('sensor_yaw').value)
        self._body_to_base_x = float(self.get_parameter('body_to_base_link_x').value)
        cfg.localization_timeout = float(self.get_parameter('localization_timeout').value)
        cfg.obstacle_timeout = float(self.get_parameter('obstacle_timeout').value)
        cfg.wheel_feedback_timeout = float(self.get_parameter('wheel_feedback_timeout').value)
        self._car_state_topic = str(self.get_parameter('car_state_topic').value)
        self._bridge_state_topic = str(self.get_parameter('bridge_state_topic').value)
        cfg.emergency_stop = bool(self.get_parameter('emergency_stop').value)
        cfg.enable_trap = bool(self.get_parameter('enable_trap').value)
        # ---- 模式仲裁参数接线（2026-11-04；名与 yaml 完全一致）----
        cfg.v_streak_threshold = float(self.get_parameter('v_streak_threshold').value)
        cfg.v_streak_count = int(self.get_parameter('v_streak_count').value)
        cfg.ack_guard_time = float(self.get_parameter('ack_guard_time').value)
        cfg.spin_reset_progress = float(self.get_parameter('spin_reset_progress').value)
        cfg.trap_exit_dist = float(self.get_parameter('trap_exit_dist').value)
        cfg.trap_exit_goal_ang_deg = float(self.get_parameter('trap_exit_goal_ang_deg').value)
        cfg.trap_exit_sector_deg = float(self.get_parameter('trap_exit_sector_deg').value)
        cfg.trap_exit_sector_range = float(self.get_parameter('trap_exit_sector_range').value)
        cfg.trap_exit_sector_max_obs = int(self.get_parameter('trap_exit_sector_max_obs').value)
        cfg.trap_max_duration = float(self.get_parameter('trap_max_duration').value)
        cfg.spin_request_timeout = float(self.get_parameter('spin_request_timeout').value)
        cfg.obstacle_max_speed = float(self.get_parameter('obstacle_max_speed').value)
        cfg.control_dt = 1.0 / float(self.get_parameter('control_rate').value)
        for key in ['safety_margin','near_goal_spin_distance','near_goal_spin_angle_deg','blocked_spin_time','spin_speed']:
            setattr(cfg,key,float(self.get_parameter(key).value))
        # ---- ★2026-11-06 用户口述逻辑参数接线 ----
        cfg.ack_hold_enable = bool(self.get_parameter('ack_hold_enable').value)
        cfg.ack_hold_v_ratio = float(self.get_parameter('ack_hold_v_ratio').value)
        cfg.ack_hold_w_ratio = float(self.get_parameter('ack_hold_w_ratio').value)
        cfg.ack_bad_count = int(self.get_parameter('ack_bad_count').value)
        cfg.ack_bad_ratio = float(self.get_parameter('ack_bad_ratio').value)
        cfg.spin_count_for_trap = int(self.get_parameter('spin_count_for_trap').value)
        cfg.spin_gap_eps = float(self.get_parameter('spin_gap_eps').value)
        cfg.stall_obs_dist = float(self.get_parameter('stall_obs_dist').value)
        cfg.stall_escape_time = float(self.get_parameter('stall_escape_time').value)
        cfg.trap_exit_far = float(self.get_parameter('trap_exit_far').value)
        cfg.trap_exit_ratio = float(self.get_parameter('trap_exit_ratio').value)
        cfg.trap_exit_ang_deg = float(self.get_parameter('trap_exit_ang_deg').value)
        cfg.trap_exit_block_max = int(self.get_parameter('trap_exit_block_max').value)
        cfg.road_base_len = float(self.get_parameter('road_base_len').value)
        cfg.road_base_ds = float(self.get_parameter('road_base_ds').value)

        # ---------------- 算法核心（与仿真同构） ----------------
        pp = module_config('v15_latesc')
        pp.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        # 用户指定：固定预测时域（traj_predict_mode='fixed'，sim_time=3.0s 恒定）
        pp.traj_predict_mode = str(self.get_parameter('traj_predict_mode').value)
        pp.sim_time = float(self.get_parameter('sim_time').value)
        # APF 自转目标角参数接线（yaml 可覆盖；默认与 params.py L171-173 一致）
        pp.apf_k_att = float(self.get_parameter('apf_k_att').value)
        pp.apf_k_rep = float(self.get_parameter('apf_k_rep').value)
        pp.apf_d0 = float(self.get_parameter('apf_d0').value)
        # C8 A* 需要全局栅格，实车默认关闭（与仿真非走廊场景一致）
        pp.enable_c8_global_guide = False
        # 规划采样窗口与执行层对齐：v_max/v_min 用 yaml 值（首跑低速 0.4），
        # 避免"规划到 1.5、执行被压到 0.4"的评分失真（C 方案参数对齐）。
        pp.max_linear_velocity = float(self.get_parameter('v_max').value)
        pp.min_linear_velocity = float(self.get_parameter('v_min').value)

        fws = FourWheelSteerParams()
        fws.length = float(self.get_parameter('vehicle_length').value)
        fws.width = float(self.get_parameter('vehicle_width').value)
        fws.v_max = float(self.get_parameter('v_max').value)
        fws.v_min = float(self.get_parameter('v_min').value)
        fws.omega_max = float(self.get_parameter('omega_max').value)
        fws.omega_spin_max = float(self.get_parameter('omega_spin_max').value)
        fws.a_max = float(self.get_parameter('a_max').value)
        fws.dec_max = float(self.get_parameter('dec_max').value)
        fws.omega_acc = float(self.get_parameter('omega_acc').value)
        fws.delta_max = math.radians(float(self.get_parameter('delta_max_deg').value))

        # ---- A 方案：阿克曼运动学（默认）vs 旧 4WIS ----
        kinematics_mode = str(self.get_parameter('kinematics_mode').value)
        if kinematics_mode != 'ackermann':raise ValueError('Real-time vehicle stack supports calibrated ACK 0 / SPIN 4 only')
        if kinematics_mode == 'ackermann':
            ack = AckermannParams(
                length=float(self.get_parameter('ack_length').value),
                width=float(self.get_parameter('ack_width').value),
                wheelbase=float(self.get_parameter('ack_wheelbase').value),
                delta_max=math.radians(float(self.get_parameter('ack_delta_max_deg').value)),
                delta_dot_max=math.radians(float(self.get_parameter('ack_delta_dot_max_deg').value)),
                v_max=float(self.get_parameter('ack_v_max').value),
                v_min=float(self.get_parameter('ack_v_min').value),
                a_max=float(self.get_parameter('ack_a_max').value),
                dec_max=float(self.get_parameter('ack_dec_max').value),
            )
            # rectangle 模式下启用 Reeds-Shepp 多点掉头兜底（U 型陷阱/窄巷掉头）
            pp.enable_mpt_escape = True
            planner = RealtimeAckermannPlanner(pp, 'ENHANCED_DWA', collision_model='rectangle',
                                  ack_params=ack, fws_params=fws)
            self.get_logger().info(
                f'[A 方案] kinematics_mode=ackermann: L={ack.wheelbase}m '
                f'δ_max={math.degrees(ack.delta_max):.0f}° '
                f'κ_max={ack.kappa_max:.3f}/m R_min={ack.r_min:.3f}m '
                f'v∈[{ack.v_min},{ack.v_max}]m/s 撞检矩形 {ack.length}×{ack.width}m '
                f'mpt_escape={pp.enable_mpt_escape}')
        else:
            planner = EnhancedDWA(pp, 'ENHANCED_DWA', collision_model='rect4wis',
                                  fws_params=fws)
            self.get_logger().info(
                f'[4WIS 回退] kinematics_mode={kinematics_mode}（原地自转仍可用）')
        for key in ['planning_budget_s','safety_margin','heading_weight','steer_weight','steer_change_weight','guide_extra','command_acc','cruise_velocity','lateral_acc_max']:
            setattr(pp,key,float(self.get_parameter(key).value))
        pp.velocity_horizon_targets=bool(self.get_parameter('velocity_horizon_targets').value)
        for key in ['near_goal_spin_distance','near_goal_spin_angle_deg','blocked_spin_time','spin_speed','command_acc','command_steer_rate_deg','safety_margin','goal_replay_epsilon','motion_stall_time','recovery_subgoal_distance','recovery_max_retries','goal_max_spin_requests']:
            setattr(cfg,key,self.get_parameter(key).value)
        pp.command_steer_rate=math.radians(float(self.get_parameter('command_steer_rate_deg').value))
        self._async_planner = AsyncPlanner(planner) if bool(self.get_parameter('async_planning').value) else None
        if self._async_planner is not None: planner=self._async_planner
        self.planner = planner

        # ---- 参考轨迹发布参数（任务书第 2 节；与 ack_model 同源，_publish_traj 使用）----
        self._ack_wheelbase = float(self.get_parameter('ack_wheelbase').value)
        self._ack_delta_max_deg = float(self.get_parameter('ack_delta_max_deg').value)
        self._traj_dt = float(self.get_parameter('traj_dt').value)        # 积分步长 s（3.0s 时域 → 60 点）
        self._traj_sim_time = float(self.get_parameter('sim_time').value)
        self._traj_v_eps = float(self.get_parameter('traj_v_eps').value)  # |v| 低于此视为静止（与 AckermannParams.v_eps 一致）

        # ---- 方案B 时序缓存已移除：路基由 core 逐周期积分发布，融合在跟踪层完成 ----
        # 规划层平滑/黏滞参数接线（params.py 提供默认值，此处运行时以 yaml 覆盖）
        self.planner.params.alpha_smooth = float(self.get_parameter('smooth_weight').value)
        self.planner.params.sticky_eps = float(self.get_parameter('sticky_eps').value)
        # 路基静止阈值接线：VehicleBridgeCore._build_road_base 读 planner.params.traj_v_eps；
        # 不显式赋值时 yaml 的 traj_v_eps 不生效（会静默退化成 0.05 默认）。
        self.planner.params.traj_v_eps = self._traj_v_eps

        # ---------------- DQN 模型版注入（可选；与 eval_dqn_v5.py 同语义） ----------------
        model_path = str(self.get_parameter('model_path').value)
        if model_path:raise ValueError('Bounded non-model planner requires model_path empty')
        self.get_logger().info(f'Bounded Ackermann DWA: horizon={pp.sim_time}s, physical footprint={ack.length}x{ack.width}m')

        predictor = ObstaclePredictor(dt=cfg.control_dt,
                                      max_speed=cfg.obstacle_max_speed)

        self.core = RealtimeVehicleBridgeCore(planner, pp, fws_params=fws,
                                      predictor=predictor, config=cfg,
                                      ack_model=AckermannModel(ack) if kinematics_mode == 'ackermann' else None)

        # ---------------- 障碍物融合 ----------------
        self.clusterer = Clusterer(
            gate=float(self.get_parameter('cluster_gate').value),
            min_points=int(self.get_parameter('cluster_min_points').value),
            max_radius=float(self.get_parameter('cluster_max_radius').value),
        )
        self._scan_cache = None
        self._scan_stamp = 0.0

        # ---------------- 订阅 ----------------
        qos_volatile = QoSProfile(depth=5,
                                  reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.VOLATILE,
                                  history=HistoryPolicy.KEEP_LAST)
        if ObstacleArray is not None:
            self._sub_obs = self.create_subscription(
                ObstacleArray, '/rda/obstacles', self._cb_obstacles, qos_volatile)
        # self._sub_scan = self.create_subscription(
        #     LaserScan, '/scan', self._cb_scan, qos_volatile)
        self._sub_odom = self.create_subscription(
            Odometry, '/localization', self._cb_odom, qos_volatile)
        self._sub_pose = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._cb_amcl, qos_volatile)
        self._sub_goal = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self._cb_goal, qos_volatile)
        # ---- 底盘轮速反馈（/car_state，四轮 m/s；供 vw 采样/加速度基准）----
        if CarState is not None:
            self._sub_car_state = self.create_subscription(
                CarState, self._car_state_topic, self._cb_car_state, qos_volatile)
        # ---- 控制桥执行状态（/car_bridge_state：0=ACK 1=挂起 2=SPIN_RUN）----
        self._sub_session_state=self.create_subscription(String,'/rda/session_state',self._cb_session_state,qos_volatile)
        self._sub_bridge_state = self.create_subscription(
            Int32, self._bridge_state_topic, self._cb_bridge_state, qos_volatile)
        # ---- tracker 实际下发指令（/car_cmd，50Hz）→ V-streak 仲裁计数核心源 ----
        if CarCtr is not None:
            self._sub_car_cmd = self.create_subscription(
                CarCtr, '/car_cmd', self._cb_car_cmd, qos_volatile)
        # ---- GPS 定位（NavSatFix；以首帧为原点转局部平面坐标）----
        self._gps_origin = None        # (lat0, lon0)
        self._gps_prev_xy = None       # 上一帧局部 xy（差分估航向）
        self._gps_yaw = None           # 外部航向话题缓存（Float32, rad）
        _gps_topic = str(self.get_parameter('gps_topic').value)
        _gps_yaw_topic = str(self.get_parameter('gps_yaw_topic').value)
        if _gps_topic:
            self._sub_gps = self.create_subscription(
                NavSatFix, _gps_topic, self._cb_gps, qos_volatile)
        if _gps_yaw_topic:
            self._sub_gps_yaw = self.create_subscription(
                Float32, _gps_yaw_topic, self._cb_gps_yaw, qos_volatile)

        # ---------------- 发布 ----------------
        self._pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self._pub_steer = self.create_publisher(Float32MultiArray, '/rda/wheel_steer', 10)
        self._pub_wspd = self.create_publisher(Float32MultiArray, '/rda/wheel_speed', 10)
        self._pub_status = self.create_publisher(String, '/rda/status', 10)
        # ---- 预测轨迹跟踪改造（任务书）：固定时域参考轨迹 + APF 自转目标角 + 模式命令 ----
        # /rda/traj: nav_msgs/Path（map 系，base_link 口径，60 点，3.0s 固定时域恒速积分）
        # /rda/spin_target: std_msgs/Float32（rad，map 系）＝ APF 力方向（自转目标角）
        # /rda/traj_mode: std_msgs/Int32（0=纯跟踪 ACK，1=请求自转 SPIN；2026-11-04 起为
        #   planner 侧仲裁产出的控制命令，tracker 订阅消费执行；不再是诊断量）
        self._pub_plan = self.create_publisher(String, '/rda/plan', 5)
        self._pub_traj = self.create_publisher(Path, '/rda/traj', 10)
        self._pub_traj_clear = self.create_publisher(Int32, '/rda/traj_clear', 10)
        self._pub_spin_target = self.create_publisher(Float32, '/rda/spin_target', 10)
        self._pub_traj_mode = self.create_publisher(Int32, '/rda/traj_mode', 10)
        # ---- 【诊断只读】2026-11-03：/rda/diag String(JSON) trap 诊断量发布 ----
        # 纯观测：每规划帧把 trap 状态机内部量（stall/子目标/escape_heading/freeze_stall
        # 等）序列化为 JSON 文本发布，供 diag_recorder_node 落地。不改任何控制行为。
        self._pub_diag = self.create_publisher(String, '/rda/diag', 10)
        self._last_traj_arc = 0.0   # _publish_traj 计算的最新轨迹弧长（m，诊断用）
        # P2-修复②：SPIN_REQUEST 期间锁存的自转目标角（进入请求时快照，防每帧 APF 重算漂移）
        self._spin_target_latched = 0.0
        self._spin_req_active = False
        self._last_clear_seq = -1     # 最近已发布的 /rda/traj_clear 序号（去重）
        self._last_theta_target = 0.0  # 最近下发的自转目标角（诊断）
        self._last_theta_apf = 0.0     # 最近算出的 APF 合力方向（诊断）
        self._spin_target_free_fallback = bool(
            self.get_parameter('spin_target_free_fallback').value)

        # ---------------- 定时主循环 ----------------
        rate = float(self.get_parameter('control_rate').value)
        self._timer = self.create_timer(1.0 / rate, self._control_cycle)

        # 初始化时间基线（用节点时间）
        self.core._t_now = 0.0
        self.core._t_loc = -1e9
        self.core._t_obs = -1e9
        # ---- C 方案：真实周期统计（dt 用实测间隔，M1 动态窗口/障碍预测与墙钟对齐）----
        self._last_cycle_t = None
        self._cyc_cnt = 0
        self._cyc_sum = 0.0
        self._cyc_max = 0.0
        self.get_logger().info(
            f'rda_dwa_real_bridge 启动: algorithm_frame={cfg.algorithm_frame} '
            f'rate={rate}Hz trap={cfg.enable_trap} 急停={cfg.emergency_stop}')

    # ==================================================================
    # 输入回调
    # ==================================================================
    def _cb_obstacles(self, msg):
        """结构化障碍物消息 → 算法坐标系检测列表。"""
        stamp = self._stamp_sec(msg.header)
        dets = []
        for ob in msg.obstacles:
            otype = ob.type if ob.type in ('pedestrian', 'bicycle', 'vehicle',
                                           'static_large', 'block') else 'unknown'
            dets.append((ob.x, ob.y, otype, ob.length, ob.width,
                         ob.heading, ob.speed, ob.id, ob.shape_heading if getattr(ob,'has_shape_heading',False) else ob.heading))
        self.core.set_obstacles(dets, stamp, frame=msg.header.frame_id, arrival=time.monotonic(),source_age=max(0.,self.get_clock().now().nanoseconds*1e-9-stamp))

    # def _cb_scan(self, msg):
    #     """LaserScan → 聚类 → 障碍物检测（坐标系 = msg.header.frame_id）。"""
    #     stamp = self._stamp_sec(msg.header)
    #     points = []
    #     angle = msg.angle_min
    #     n = len(msg.ranges)
    #     for i in range(n):
    #         r = msg.ranges[i]
    #         if not math.isfinite(r) or r <= msg.range_min:
    #             angle += msg.angle_increment
    #             continue
    #         if r >= msg.range_max:
    #             angle += msg.angle_increment
    #             continue
    #         # 传感器系（极坐标 → 笛卡尔）
    #         points.append((r * math.cos(angle), r * math.sin(angle)))
    #         angle += msg.angle_increment
    #     dets = []
    #     for cl in self.clusterer.cluster(points):
    #         cx, cy, L, W, theta = cluster_bounding_box(cl)
    #         otype = infer_type(L, W)
    #         dets.append((cx, cy, otype, L, W, theta, 0.0, None))
    #     self._scan_cache = dets
    #     self._scan_stamp = stamp
    #     self.core.set_obstacles(dets, stamp, frame=msg.header.frame_id, arrival=time.monotonic())

    def _cb_odom(self, msg):
        """/odom：定位位姿（frame=odom，须与 algorithm_frame 一致或经 TF）。

        /localization 给的是 body 在 map 下的位姿；算法消费 base_link 位姿，
        需在此处做 body→base_link 2D 偏移补偿。
        """
        if self._localization_input.uses_tf:
            return
        p = msg.pose.pose
        x_body = p.position.x
        y_body = p.position.y
        yaw_body = quat_to_yaw(p.orientation.x, p.orientation.y,
                               p.orientation.z, p.orientation.w)
        x_base, y_base, yaw_base = body_pose_to_base_link(
            x_body, y_body, yaw_body, self._body_to_base_x)
        self.core.set_localization(
            x_base, y_base, yaw_base,
            self._stamp_sec(msg.header), arrival=time.monotonic())
        # 若存在速度反馈，可更新 state（此处保留 state 由模型积分）

    def _cb_amcl(self, msg):
        """/amcl_pose：AMCL 定位（frame=map，须与 algorithm_frame 一致或经 TF）。

        与 _cb_odom 同理：amcl 位姿为 body 系，需补偿为 base_link。
        """
        if self._localization_input.uses_tf:
            return
        p = msg.pose.pose
        x_body = p.position.x
        y_body = p.position.y
        yaw_body = quat_to_yaw(p.orientation.x, p.orientation.y,
                               p.orientation.z, p.orientation.w)
        x_base, y_base, yaw_base = body_pose_to_base_link(
            x_body, y_body, yaw_body, self._body_to_base_x)
        self.core.set_localization(
            x_base, y_base, yaw_base,
            self._stamp_sec(msg.header), arrival=time.monotonic())

    def _cb_goal(self, msg):
        """/goal_pose：目标点。frame 为算法系直接用；车体系按定位转换。"""
        stamp = self._stamp_sec(msg.header)
        gx, gy = msg.pose.position.x, msg.pose.position.y
        frame = msg.header.frame_id or self.core.cfg.algorithm_frame
        self.get_logger().info(
            f"[DIAG] goal recv x={gx:.3f} y={gy:.3f} frame={frame} stamp={stamp:.3f}")
        if is_body_frame(frame, self.core.cfg.body_frames):
            # 目标点给在车体系（少见）→ 按当前定位转到算法系
            from .frame_utils import body_to_world_2d
            gx, gy, _ = body_to_world_2d(
                gx, gy, 0.0, *self.core.centre(),
                self.core.robot.theta)
            frame = self.core.cfg.algorithm_frame
        self.core.set_goal(gx, gy, stamp, frame=frame)

    def _cb_car_state(self, msg):
        """底盘四轮轮速反馈（CarState.speed[4]，m/s）→ 规划层 v/ω 融合（见 vehicle_bridge）。"""
        self.core.set_wheel_feedback(list(msg.speed), stamp=time.monotonic())
        self.core.set_mode_ready(int(msg.turnmode), arrival=time.monotonic())

    def _cb_session_state(self,msg):
        try:
            packet=json.loads(msg.data)
            self.core.set_bridge_state(int(packet['state']),session=int(packet['session']))
        except (ValueError,TypeError,KeyError):pass

    def _cb_bridge_state(self, msg):
        """控制桥执行状态（Int32：0=ACK 1=挂起 2=自转执行中）→ trap stall 冻结/vw 反馈来源。"""
        self.core.set_bridge_state(int(msg.data))

    def _cb_car_cmd(self, msg):
        """tracker 实际下发速度（CarCtr.speed，m/s，50Hz）→ V-streak 仲裁计数核心源。

        2026-11-04：仲裁上收 planner 后，此处把 tracker 真实下发线速度喂给
        VehicleBridgeCore.set_tracker_cmd_speed（ACK 门控内累计低速帧）。"""
        self.core.set_tracker_cmd_speed(float(msg.speed), float(msg.angle), int(msg.turnmode), arrival=time.monotonic())

    # ==================================================================
    # GPS 定位回调（NavSatFix → 局部平面坐标；以首帧为原点）
    # ==================================================================
    @staticmethod
    def _gps_to_local(lat, lon, lat0, lon0):
        """等距圆柱近似：小范围内经纬度增量 → 米（x=东向, y=北向）。"""
        import math as _m
        _R_E = 6378137.0
        x = _R_E * _m.radians(lon - lon0) * _m.cos(_m.radians(lat0))
        y = _R_E * _m.radians(lat - lat0)
        return x, y

    def _cb_gps(self, msg):
        """/fix（NavSatFix）→ 局部 xy → set_localization。"""
        if not (msg.status and msg.status.status >= 0):
            # 无效 fix（如 0=未固定/1=单点等；status>=0 视为有可用解）——保守起见仅接受 status>=0
            # 注：status 枚举 0=NO_FIX,1=FIX,2=SBAS；此处按"非 -1 即视为有 fix"放宽
            pass
        lat = float(msg.latitude)
        lon = float(msg.longitude)
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return
        if self._gps_origin is None:
            self._gps_origin = (lat, lon)
            self._gps_prev_xy = None
        lat0, lon0 = self._gps_origin
        x, y = self._gps_to_local(lat, lon, lat0, lon0)

        # 航向：优先外部航向话题；否则 GPS 位移差分（仅运动时有效）
        theta = self._gps_yaw if self._gps_yaw is not None else None
        if theta is None and self.get_parameter('gps_yaw_use_diff').value \
                and self._gps_prev_xy is not None:
            dx, dy = x - self._gps_prev_xy[0], y - self._gps_prev_xy[1]
            if math.hypot(dx, dy) > 1e-3:
                theta = math.atan2(dy, dx)
        self._gps_prev_xy = (x, y)
        if theta is None:
            theta = 0.0
        self.core.set_localization(x, y, theta, self._stamp_sec(msg.header))

    def _cb_gps_yaw(self, msg):
        """外部航向话题（std_msgs/Float32，rad，逆时针为正）。"""
        self._gps_yaw = float(msg.data)

    # ==================================================================
    # 主循环
    # ==================================================================
    def _control_cycle(self):
        cfg = self.core.cfg
        # ---- C 方案：用真实执行周期 dt（墙钟实测），不再用固定 1/rate ----
        # 依据：监控 CSV 实测 /cmd_vel 实际仅 ~5-7Hz（timer 20Hz 但 plan 耗时 0.1~0.9s），
        # 固定 dt=0.05 导致每条指令被底盘执行 4~17 倍时长 → 过冲 → 来回修正（爬行）。
        # 用真实 dt：M1 动态窗口（acc·dt）与障碍预测按墙钟步进，规划周期失配消除。
        now = time.monotonic()
        if self._last_cycle_t is None:
            dt_real = cfg.control_dt
        else:
            dt_real = max(0.005, min(0.5, now - self._last_cycle_t))
        raw_cycle_dt=cfg.control_dt if self._last_cycle_t is None else now-self._last_cycle_t
        self._last_cycle_t = now
        # 算法 dt 同步为真实周期（M1 窗口/预测时域同源）
        self.planner.params.dt = dt_real

        sample = self._localization_input.sample(cfg.localization_timeout)
        if sample is not None:
            pose, stamp, age = sample
            x, y, yaw = body_pose_to_base_link(*pose, self._body_to_base_x)
            self.core.set_localization(x, y, yaw, stamp, arrival=now-age)

        t0 = time.monotonic()
        info = self.core.step(dt=dt_real, now=now)
        dur = time.monotonic() - t0
        info['timing']=dict(supervisor_monotonic_ns=time.monotonic_ns(),raw_cycle_dt_s=raw_cycle_dt,used_cycle_dt_s=dt_real,step_duration_s=dur)


        # 周期/耗时统计（每 50 帧输出一次，佐证 C 方案降频依据）
        self._cyc_cnt += 1
        self._cyc_sum += dur
        self._cyc_max = max(self._cyc_max, dur)
        if self._cyc_cnt % 50 == 0:
            avg = self._cyc_sum / self._cyc_cnt
            self.get_logger().info(
                f"[C 方案] 近{self._cyc_cnt}帧: plan均值={avg*1e3:.0f}ms "
                f"最大={self._cyc_max*1e3:.0f}ms 有效周期≈{1.0/avg:.1f}Hz "
                f"dt_real≈{self._cyc_sum/self._cyc_cnt*1e3:.0f}ms")
            self._cyc_cnt = 0
            self._cyc_sum = 0.0
            self._cyc_max = 0.0

        # ---- 发布控制指令 ----
        twist = Twist()
        twist.linear.x = info['v_cmd']
        twist.angular.z = info['w_cmd']
        self._pub_cmd.publish(twist)
        # self.get_logger().info(
        #     f"[DIAG] PUB /cmd_vel v={info['v_cmd']:.3f} w={info['w_cmd']:.3f} "
        #     f"reason={info.get('reason', '-')} plan_ok={info.get('plan_ok', '?')}")

        # ---- 预测轨迹跟踪改造：发布固定时域参考轨迹 + APF 自转目标角 + 模式诊断 ----
        # 频率 = 规划频率（5~7Hz），随主循环发布，不用额外 timer（任务书第 2 节）。
        self._publish_traj(info)

        # ---- 【诊断只读】发布 /rda/diag（trap 内部量 JSON） ----
        self._publish_diag(info)

        # ---- 四轮转向/轮速（4WIS 输出） ----
        steers, speeds = self.core.wheel_commands()
        sm = Float32MultiArray()
        sm.data = [float(s) for s in steers]
        self._pub_steer.publish(sm)
        wm = Float32MultiArray()
        wm.data = [float(s) for s in speeds]
        self._pub_wspd.publish(wm)

        # ---- 状态诊断 ----
        status = String()
        st = info.get('reason', '')
        dg = info['dist_goal']
        dg_str = f'{dg:.2f}' if math.isfinite(dg) else '-'
        status.data = (
            f"t={self.core._t_now:.1f} v={info['v_cmd']:.2f} w={info['w_cmd']:.2f} "
            f"ok={int(info['ok'])} obs={info['n_obs']} "
            f"trap={int(info['trap_active'])} coll={int(info['collision'])} "
            f"reached={int(info['reached'])} dg={dg_str}"
            + (f" reason={st}" if st else ""))
        self._pub_status.publish(status)

    def _publish_traj(self, info: dict):
        """发布"前方 2.5m 路基" + 清空路径命令 + APF 自转目标角 + 模式命令。

        1) /rda/traj（nav_msgs/Path）—— 路基由 VehicleBridgeCore 按"当前位姿 + 最终 (ω,v)"
           积分（road_base_len=2.5m、点间距 road_base_ds=0.1m），每点携带三要素：
             position    = (x, y, v_ref)     z 位放该点参考速度 m/s（跟踪层 v_ref 来源）
             orientation = θ_ref 的 yaw 四元数（LQR 横向/航向误差的参考系）
             header.stamp= now + t_rel       该点参考到达时刻（跟踪层按时间 t 插值查询）
           自转模式 / 停车（|v|<v_eps）→ 发**空 Path**（跟踪层无路基可跟，只执行转向）。
        2) /rda/traj_clear（Int32，序号递增）—— 模式切换 / trap 激活 / trap 退出瞬间发一次，
           跟踪层收到后清空其参考滑窗（旧路基已不对应当前模式/目标）。
        3) /rda/spin_target（Float32）—— 顶层=当前位姿 APF 合力方向；trap 内=冻结子目标方向。
        4) /rda/traj_mode（Int32）—— 0=阿克曼跟踪 1=自转（模式唯一决策来源）。
        """
        v = float(info.get('v_cmd', 0.0))
        w = float(info.get('w_cmd', 0.0))
        x = float(self.core.robot.x)
        y = float(self.core.robot.y)
        theta = float(self.core.robot.theta)

        # ---- 1) 路基发布：点 = (x, y, θ, v, t_rel)，弧长 2.5m、间距 0.1m ----
        # 路基由 VehicleBridgeCore._build_road_base 用"当前位姿 + 最终 (ω,v)"积分得到
        # （与 predict_reference_traj 的定时域口径不同：这里按里程截断）。
        rb = info.get('road_base') or []
        now_msg = self.get_clock().now().to_msg()
        t0 = float(now_msg.sec) + float(now_msg.nanosec) * 1e-9
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = now_msg
        for (px, py, pth, pv, pt) in rb:
            ps = PoseStamped()
            tt = t0 + float(pt)
            ps.header.frame_id = 'map'
            ps.header.stamp.sec = int(math.floor(tt))
            ps.header.stamp.nanosec = int((tt - math.floor(tt)) * 1e9)
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.position.z = float(pv)     # z 位 = 该点参考速度（跟踪层 LQR 的 v_ref）
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = math.sin(0.5 * float(pth))   # θ_ref 的 yaw 四元数
            ps.pose.orientation.w = math.cos(0.5 * float(pth))
            path.poses.append(ps)
        self._pub_traj.publish(path)          # 自转/停车时 rb=[] → 发空 Path（无路基）

        # ---- 诊断用：路基弧长（m；空路基 → 0）----
        arc = 0.0
        for i in range(1, len(rb)):
            arc += math.hypot(rb[i][0] - rb[i - 1][0], rb[i][1] - rb[i - 1][1])
        self._last_traj_arc = float(arc)

        # ---- 1b) 清空跟踪路径命令（模式切换 / trap 激活 / trap 退出瞬间，序号递增）----
        seq = info.get('traj_clear_seq')
        if seq is not None and int(seq) != self._last_clear_seq:
            self._last_clear_seq = int(seq)
            self._pub_traj_clear.publish(Int32(data=int(seq)))
            self.get_logger().info(f'[DIAG] CLEAR_PATH /rda/traj_clear seq={seq}')

        # Atomic versioned command is authoritative for the tracker. Legacy
        # Path/mode/target topics remain diagnostics; DDS order cannot split a session.
        theta_target=float(info.get('spin_target',theta))
        self._last_theta_apf=self._last_theta_target=theta_target
        self._pub_spin_target.publish(Float32(data=theta_target))
        self._pub_plan.publish(String(data=json.dumps({
            'stamp':time.monotonic(),'session':int(info.get('spin_session',0)),
            'mode':int(info.get('traj_mode',0)),'target':theta_target,
            'reason':info.get('reason','fault'),'reference':rb,
            'planner':info.get('planner',{}),
        },allow_nan=False)))

        # ---- 3) 模式命令（/rda/traj_mode：0=纯跟踪 ACK，1=请求 tracker 自转 SPIN） ----
        # 2026-11-04：旧诊断语义（|v|<0.05 且 |w|>0.10）废弃删除；现由 planner 侧仲裁
        # 状态机（VehicleBridgeCore._arbitration_step）产出 info['traj_mode']，
        # tracker 订阅消费后执行其 HOLD→RUN→reached/超时 FSM（另一 agent 同步改造）。
        self._pub_traj_mode.publish(Int32(data=int(info.get('traj_mode', 0))))

    def _publish_diag(self, info: dict):
        """【诊断只读】每规划帧发布 /rda/diag（std_msgs/String，JSON 文本）。

        字段（全部只读快照，不参与任何控制分支）：
          t              核心时钟秒（self.core._t_now）
          stall          trap 位移停滞计数（trap_ctrl.stall）
          stall_run      已连续静止帧数（trap_ctrl.stall_run）
          trap_steps     trap 状态机步数（trap_ctrl.trap_steps）
          trap_active    trap 激活标志
          sub_goal_x/y   trap 逃逸子目标（map 系，米；无则 None）
          escape_heading trap 逃逸目标航向（rad；无则 None）
          freeze_stall   本帧 stall 是否被执行状态冻结（bridge_state in (1,2)）
          bridge_state   执行状态 0=ACK 1=HANGUP 2=SPIN_RUN
          traj_arc_len   最新预测轨迹弧长（m；|v|≈0 时 ≈0）
        """
        tc = self.core.trap_ctrl
        trap = self.core.trap
        sub = info.get('recovery_goal')
        try:
            sub_x = float(sub[0]) if sub is not None else None
            sub_y = float(sub[1]) if sub is not None else None
        except Exception:
            sub_x, sub_y = None, None
        diag = {
            't': float(self.core._t_now),
            'reason': info.get('reason',''),
            'localization_input': self._localization_input.diagnostic(),
            'timing':info.get('timing'),
            'pose':dict(centre=self.core.centre(),rear=[self.core.robot.x,self.core.robot.y,self.core.robot.theta],source_stamp=self.core._pose_source,arrival_age_s=self.core._t_now-self.core._t_loc,v_estimate=self.core._v_meas),
            'goal':self.core._goal,
            'goal_received':bool(self.core._goal_received),
            'obstacle_source_stamp':self.core._obs_source,
            'obstacle_age_s':self.core._t_now-self.core._t_obs,
            'planner': info.get('planner',{}),
            'spin_session':info.get('spin_session',0),
            'goal_spin_requests':info.get('goal_spin_requests',0),
            'recovery_goal':info.get('recovery_goal'),
            'recovery_failed':info.get('recovery_failed',False),
            'recovery_elapsed':info.get('recovery_elapsed',0.),
            'stall': getattr(tc, 'stall', 0),
            'stall_run': getattr(tc, 'stall_run', 0.0),
            'trap_steps': getattr(tc, 'trap_steps', 0),
            'trap_active': bool(info.get('trap_active', False)),
            'sub_goal_x': sub_x,
            'sub_goal_y': sub_y,
            'escape_heading': (float(trap.get('escape_heading'))
                                if isinstance(trap, dict)
                                and trap.get('escape_heading') is not None else None),
            'freeze_stall': info.get('reason') not in ('motion','blocked','orientation_stall'),
            'bridge_state': int(self.core._bridge_state),
            'traj_arc_len': float(self._last_traj_arc),
            # ---- 模式仲裁诊断（2026-11-04）----
            'arbit': str(info.get('arbit', '?')),        # ACK / SPIN_REQUEST
            'v_streak': int(info.get('v_streak', 0)),   # 当前低速连续帧数
            'traj_mode': int(info.get('traj_mode', 0)),  # 实际下发模式命令
            # ★2026-11-06：卡顿计时与 ω 基准反馈（无轮速时取定位 θ 差分）
            'stall_time': float(info.get('stall_time', 0.0)),
            'omega_ref': float(getattr(self.core.robot, 'omega', 0.0)),
            # ---- ★2026-11-06 用户口径仲裁量（缺一项就无法复盘"为何没切自转/没进 trap"）----
            'ack_bad_cnt': int(info.get('ack_bad_cnt', 0)),   # 连续不可行帧数（阈值 5）
            'spin_count': int(info.get('spin_count', 0)),     # 自转累计激活次数（≥2 才 trap）
            'spin_gap': float(self.core._spin_gap()),
            'spin_target': float(self._last_theta_target),
            'spin_target_apf': float(self._last_theta_apf),
            'clear_seq': int(info.get('traj_clear_seq', 0)),  # 清空跟踪路径命令序号
            'road_base_len': float(self._last_traj_arc),      # 下发路基弧长（0=自转/停车）
            'trap_sub_dist0': float(getattr(self.core, '_trap_sub_dist0', 0.0)),
        }
        self._pub_diag.publish(String(data=json.dumps(safe_data(diag), ensure_ascii=False, allow_nan=False)))

    def destroy_node(self):
        if self._async_planner is not None: self._async_planner.close()
        return super().destroy_node()

    @staticmethod
    def _stamp_sec(header) -> float:
        try:
            return header.stamp.sec + header.stamp.nanosec * 1e-9
        except Exception:
            return 0.0


def main(args=None):
    rclpy.init(args=args)
    node = RealVehicleBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
