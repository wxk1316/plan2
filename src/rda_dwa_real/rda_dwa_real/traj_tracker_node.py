"""50 Hz geometric rear-axle tracker and session-based body-centre spin.

Authoritative /rda/plan JSON contains one atomic mode/target/reference/version.
Localization remains the body centre. Geometry is converted inside the core.
Latest obstacle rectangles guard the final rate-limited command. Session results
stay latched on /rda/session_state until the supervisor returns a mode request.
Legacy paths/mode/target subscriptions are enabled only with use_atomic_plan=False.
"""
from __future__ import annotations

import math
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String, Int32, Float32
try:
    from control_msgs.msg import CarCtr
except Exception:
    CarCtr = None  # 未 build 前允许语法检查

from .telemetry_recording import safe_data
from .traj_tracker_core import TrajTrackerCore, TrajTrackerParams, wrap_angle
from .frame_utils import quat_to_yaw, body_pose_to_base_link
from .localization_input import LocalizationInput
from rda_dwa.motion_safety import TrackedObstacle
from rda_dwa_real_msgs.msg import ObstacleArray
from control_msgs.msg import CarState


# 数值参数表（名字 → 取值来源=TrajTrackerParams 属性；全部 float）
_NUM_PARAMS = [k for k,v in vars(TrajTrackerParams()).items() if isinstance(v,(float,int))]


class TrajTrackerNode(Node):
    """路基（带时间戳）→ LQR 转角 + PID 速度 → /car_cmd（固定 50Hz）。"""

    def __init__(self):
        super().__init__('traj_tracker_node')
        self._localization_input = LocalizationInput(self)
        pp = TrajTrackerParams()

        # ---------------- 数值参数 ----------------
        for name in _NUM_PARAMS:
            self.declare_parameter(name, float(getattr(pp, name)))
        # ---------------- 话题/常量参数 ----------------
        self.declare_parameter('body_to_base_link_x', 0.0)   # 与 bridge_node 同一补偿
        self.declare_parameter('rate', 50.0)                   # 控制频率 Hz
        self.declare_parameter('traj_topic', '/rda/traj')
        self.declare_parameter('traj_clear_topic', '/rda/traj_clear')
        self.declare_parameter('traj_mode_topic', '/rda/traj_mode')
        self.declare_parameter('spin_target_topic', '/rda/spin_target')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('localization_topic', '/localization')
        self.declare_parameter('car_cmd_topic', '/car_cmd')
        self.declare_parameter('status_topic', '/rda/ctrl_status')
        self.declare_parameter('bridge_state_topic', '/car_bridge_state')

        for name in _NUM_PARAMS:
            setattr(pp, name, float(self.get_parameter(name).value))
        self.core = TrajTrackerCore(pp, on_transition=self._on_mode_transition,
                                    on_clear=self._on_clear)

        self._rate = float(self.get_parameter('rate').value)
        self._body_to_base_x = float(self.get_parameter('body_to_base_link_x').value)
        self._traj_topic = str(self.get_parameter('traj_topic').value)
        self._clear_topic = str(self.get_parameter('traj_clear_topic').value)
        self._traj_mode_topic = str(self.get_parameter('traj_mode_topic').value)
        self._spin_target_topic = str(self.get_parameter('spin_target_topic').value)
        self._cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self._loc_topic = str(self.get_parameter('localization_topic').value)
        self._car_cmd_topic = str(self.get_parameter('car_cmd_topic').value)
        self._status_topic = str(self.get_parameter('status_topic').value)
        self._state_topic = str(self.get_parameter('bridge_state_topic').value)

        # ---------------- 订阅/发布 ----------------
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST)
        self.declare_parameter('use_atomic_plan',True)
        self._atomic=bool(self.get_parameter('use_atomic_plan').value)
        self._plan_seq=0;self._last_plan_stamp=-1e9;self._last_plan_reason='no_plan'
        self._sub_plan=self.create_subscription(String,'/rda/plan',self._cb_plan,qos)
        self._sub_obs=self.create_subscription(ObstacleArray,'/rda/obstacles',self._cb_obstacles,qos)
        self._sub_car_state=self.create_subscription(CarState,'/car_state',self._cb_car_state,qos)
        self._sub_traj = self.create_subscription(Path, self._traj_topic, self._cb_traj, qos)
        self._sub_clear = self.create_subscription(Int32, self._clear_topic,
                                                   self._cb_clear, qos)
        self._sub_mode = self.create_subscription(Int32, self._traj_mode_topic,
                                                  self._cb_traj_mode, qos)
        self._sub_spin = self.create_subscription(Float32, self._spin_target_topic,
                                                  self._cb_spin, qos)
        self._sub_cmd = self.create_subscription(Twist, self._cmd_vel_topic, self._cb_cmd, qos)
        self._sub_loc = self.create_subscription(Odometry, self._loc_topic, self._cb_loc, qos)

        if CarCtr is not None:
            self._pub_car = self.create_publisher(CarCtr, self._car_cmd_topic, 10)
        else:
            self._pub_car = None
            self.get_logger().error(
                'control_msgs 未找到（未 source car_ros 的 install/setup.bash），'
                '/car_cmd 发布被禁用')
        self._pub_control_diag=self.create_publisher(String,'/rda/control_diag',10)
        self._control_seq=0
        self._pub_status = self.create_publisher(String, self._status_topic, 10)
        self._pub_state = self.create_publisher(Int32, self._state_topic, 10)
        self._pub_session=self.create_publisher(String,'/rda/session_state',5)

        # ---------------- 状态缓存 ----------------
        self._traj_seq = 0
        self._last_traj_ts = -1e9
        self._last_cmd_ts = -1e9
        self._spin_target = 0.0
        self._loc = None
        self._heartbeat = 0
        self._last_tick = None
        self._tick_cnt = 0

        self.create_timer(1.0 / self._rate, self._control_tick)
        self.get_logger().info(
            f'traj_tracker(LQR) 启动: L={pp.wheelbase}m δmax={pp.max_steer_deg:.0f}° '
            f'fusion={pp.fusion_len}m window={pp.window_len}m '
            f'K=[{self.core._K[0, 0]:.3f},{self.core._K[0, 1]:.3f}] '
            f'pid=[{pp.pid_kp},{pp.pid_ki},{pp.pid_kd}] '
            f'rate={self._rate}Hz 订阅 {self._traj_topic}+{self._clear_topic}+'
            f'{self._traj_mode_topic}')

    # ==================================================================
    # 回调
    # ==================================================================
    def _now(self) -> float:
        return time.monotonic()

    def _cb_traj(self, msg: Path):
        """路基接收：每点 (x, y, v=position.z, yaw=orientation, t=header.stamp)。"""
        if self._atomic:return
        now = self._now()
        pts = []
        for ps in msg.poses:
            p = ps.pose
            yaw = quat_to_yaw(p.orientation.x, p.orientation.y,
                              p.orientation.z, p.orientation.w)
            t = float(ps.header.stamp.sec) + float(ps.header.stamp.nanosec) * 1e-9
            if t <= 0.0:
                t = now
            pts.append((float(p.position.x), float(p.position.y), yaw,
                        float(p.position.z), t))
        if len(pts) < 2:
            self.core.clear_path('empty_path')
            self._last_traj_ts = now
            return
        self._traj_seq += 1
        self._last_traj_ts = now
        self.core.set_reference(pts, now, self._traj_seq)

    def _cb_clear(self, msg: Int32):
        """/rda/traj_clear：模式切换 / trap 激活 / trap 退出 → 清空跟踪路径。"""
        if not self._atomic:self.core.consume_clear(int(msg.data))

    def _cb_traj_mode(self, msg: Int32):
        if not self._atomic:self.core.set_traj_mode(int(msg.data), self._now(), spin_target=self._spin_target)

    def _cb_spin(self, msg: Float32):
        if not self._atomic:self._spin_target = float(msg.data)

    def _cb_cmd(self, msg: Twist):
        # 仅作规划链路看门狗：2s 无 /cmd_vel → 急停（速度来自路基 v_ref，不取 /cmd_vel）
        if not self._atomic:self._last_cmd_ts = self._now()

    def _cb_loc(self, msg: Odometry):
        """/localization（body 位姿）→ base_link 补偿（与 bridge_node 同一口径）。"""
        if self._localization_input.uses_tf:
            return
        p = msg.pose.pose
        if self._localization_input.uses_latest_topic:
            self._localization_input.receive_topic(p, msg.header, received_at=self._now())
            return
        x_body = float(p.position.x)
        y_body = float(p.position.y)
        yaw_body = quat_to_yaw(p.orientation.x, p.orientation.y,
                               p.orientation.z, p.orientation.w)
        x_base, y_base, yaw_base = body_pose_to_base_link(
            x_body, y_body, yaw_body, self._body_to_base_x)
        now=self._now()
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        self.core.set_pose(x_base,y_base,yaw_base,stamp=stamp,arrival=now)
        self._loc = (x_base, y_base, yaw_base)

    def _on_mode_transition(self, mode: str, t: float, stage=None):
        self.get_logger().info(f"[DIAG] MODE->{mode} t={t:.3f}")

    def _on_clear(self, reason: str):
        self.get_logger().info(
            f"[DIAG] CLEAR_PATH reason={reason} planner_reason={getattr(self, '_last_plan_reason', 'no_plan')}")

    def _cb_plan(self,msg):
        if not self._atomic:return
        try:
            p=json.loads(msg.data);now=self._now();stamp=float(p['stamp'])
            if not math.isfinite(stamp) or stamp<=self._last_plan_stamp or now-stamp>self.core.p.cmd_timeout:return
            target=float(p['target']);session=int(p['session']);mode=int(p['mode'])
            if mode not in (0,1) or not math.isfinite(target):raise ValueError('plan fields')
            reason=str(p['reason']);self._last_plan_reason=reason
            self._last_plan_stamp=stamp;self._last_cmd_ts=now;self._spin_target=target
            healthy=reason in ('motion','goal_braking','dynamic_braking','dynamic_wait','spin_requested','spin_active','spin_reached','reached','mode_return_wait')
            self.core.set_safety('' if healthy else reason)
            self.core.set_traj_mode(mode,now,session,target)
            if mode==0:
                self._plan_seq+=1;self.core.set_reference(p.get('reference',[]),now,self._plan_seq);self._last_traj_ts=now
        except (ValueError,KeyError,TypeError):
            self._last_cmd_ts=-1e9;self.core.set_safety('invalid_plan')

    def _cb_car_state(self,msg):
        now=self._now();self.core.set_mode_ready(int(msg.turnmode),now);self.core.set_speed_feedback(list(msg.speed),now)

    def _cb_obstacles(self,msg):
        now=self._now();obs=[]
        if msg.header.frame_id not in ('map',''):
            self.core._obstacle_at=-1e9;self.core.set_safety('obstacle_frame');return
        for o in msg.obstacles:
            typ=str(o.type);speed=max(0.,float(o.speed))
            if not all(math.isfinite(float(v)) for v in (o.x,o.y,o.length,o.width,o.heading,o.speed)) or o.length<=0 or o.width<=0:
                self.core._obstacle_at=-1e9;self.core.set_safety('invalid_obstacle');return
            if typ.startswith('static') or typ in ('block','unknown'):speed=0.
            obs.append(TrackedObstacle(float(o.x),float(o.y),float(o.length),float(o.width),float(o.shape_heading) if getattr(o,'has_shape_heading',False) else float(o.heading),
                       speed*math.cos(o.heading),speed*math.sin(o.heading),int(o.id),typ=='unknown',0.))
        source=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        age=max(0.,self.get_clock().now().nanoseconds*1e-9-source)
        self.core.set_obstacles(obs,now-age,stamp=source)

    def _control_tick(self):
        begin=time.monotonic();now=self._now();raw_dt=.02 if self._last_tick is None else now-self._last_tick;dt=max(.001,min(.1,raw_dt));self._last_tick=now
        reason=''
        sample = self._localization_input.sample(self.core.p.localization_timeout, now=now)
        if sample is not None:
            pose, stamp, age = sample
            x, y, yaw = body_pose_to_base_link(*pose, self._body_to_base_x)
            self.core.set_pose(x, y, yaw, stamp=stamp, arrival=now-age,
                               allow_reinitialization=self._last_plan_reason=='no_goal')
            self._loc = (x, y, yaw)
        if now-self._last_cmd_ts>self.core.p.cmd_timeout:reason='plan_timeout'
        elif now-self.core._pose_at>self.core.p.localization_timeout:reason='localization_timeout'
        elif now-self.core._obstacle_at>self.core.p.obstacle_timeout:reason='obstacle_timeout'
        if reason:self.core.set_safety(reason)
        elif self._last_plan_reason in ('motion','goal_braking','dynamic_braking','dynamic_wait','spin_requested','spin_active','spin_reached','reached','mode_return_wait') and self.core._safety_reason not in ('invalid_localization','localization_jump'):
            self.core.set_safety('')
        cmd=self.core.step(now,dt,self._spin_target)
        self._publish(cmd)
        self._control_seq+=1
        packet=dict(schema_version=1,producer_monotonic_ns=time.monotonic_ns(),producer_ros_ns=self.get_clock().now().nanoseconds,sequence=self._control_seq,raw_dt_s=raw_dt,used_dt_s=dt,duration_s=time.monotonic()-begin,mode=self.core.mode,stage=self.core._spin_stage,session=self.core._session,result=self.core._spin_result,target_yaw_rad=self.core._spin_target,safety_reason=self.core._safety_reason,command=cmd,tracking=self.core.diag,measured_v_estimate=self.core._v_meas,yaw_rate_estimate=self.core._yaw_rate,centre=self.core._centre,rear_pose=[self.core._px,self.core._py,self.core._pyaw],pose_source_stamp=self.core._pose_stamp,pose_arrival_age_s=now-self.core._pose_at,obstacle_source_stamp=self.core._obstacle_source,obstacle_age_s=now-self.core._obstacle_at,wheel_max_cached=self.core._wheel_max,wheel_arrival_age_s=now-self.core._wheel_at,mode_echo=self.core._mode_echo,mode_echo_age_s=now-self.core._mode_echo_at,reference_seq=self.core._ref_seq,reference_age_s=now-self.core._last_traj_ts,reference_points=len(self.core._ref),steer_rate_rad_s=self.core._steer_rate,acceleration_state=self.core._acc,pid_integral=self.core._pid_i)
        packet['localization_input'] = self._localization_input.diagnostic()
        packet['pose_reinitializations']=self.core._pose_reinitializations
        self._pub_control_diag.publish(String(data=json.dumps(safe_data(packet),allow_nan=False)))

    # ==================================================================
    # 发布
    # ==================================================================
    def _publish(self, cmd: dict, extra: str = ''):
        self._pub_state.publish(Int32(data=int(self.core.bridge_state())))
        self._pub_session.publish(String(data=json.dumps({'session':self.core._session,'state':self.core.bridge_state(),'mode':self.core.mode,'result':self.core._spin_result})))
        if self._pub_car is None or CarCtr is None:
            return
        self._heartbeat = (self._heartbeat + 1) % 2147483647
        self._pub_car.publish(CarCtr(source=1,
                                     ctr_mode=1,
                                     up_down_flag=0,
                                     enable=int(cmd['enable']),
                                     turnmode=int(cmd['turnmode']),
                                     speed=float(cmd['speed']),
                                     angle=float(cmd['angle']),
                                     heartbeat=int(self._heartbeat)))
        self._pub_status.publish(String(data=self.core.status_line(extra)))
        self._tick_cnt += 1
        if self._tick_cnt % 50 == 0 or extra:
            self.get_logger().info(
                f"[DIAG] PUB /car_cmd turnmode={int(cmd['turnmode'])} "
                f"speed={float(cmd['speed']):.3f} angle={float(cmd['angle']):.3f} "
                f"mode={self.core.mode}" + (f" {extra}" if extra else ""))


def main(args=None):
    rclpy.init(args=args)
    node = TrajTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
