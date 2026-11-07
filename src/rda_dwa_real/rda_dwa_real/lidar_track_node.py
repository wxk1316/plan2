"""
lidar_track_node.py — 3D 激光点云 → 全局系障碍物跟踪 → /rda/obstacles（带稳定 id）。

数据链（与 RL-DWA 实车适配层对接）：
  /rslidar_points → 本节点内部（地面 z 过滤 + 车体自剔除）→ 二维点 (base_link 系)
       ↓ 本节点
  1) PointCloud2 → 二维点 (x, y)（rslidar 系 z 范围过滤 + 车体矩形盒自剔除）
     注：rslidar 系 (x,y) 与 base_link 系 (x,y) 完全相同（雷达在 base_link 正上方 0.45m，
     x/y 投影对齐），故车体剔除盒在 rslidar/base_link 系下语义一致。
  2) 距离聚类（复用 obstacle_fusion.Clusterer）→ 矩形包围盒 → 类型推断
  3) 用 /localization 位姿（body→base_link 补偿后）把检测从车体系(base_link)转到全局系(map)
  4) 全局系卡尔曼滤波(CV) + 匈牙利匹配（kf_tracker.GlobalKFTracker）→ 稳定 id
  5) 发布 /rda/obstacles（frame_id='map'，id/type/x/y/length/width/heading/speed）

下游：bridge_node 直接消费已跟踪的 map 障碍；不再次估计速度。
完整周围点云经体素极值保留和连通聚类；形状朝向与运动方向分别发布。

依赖：sensor_msgs_py（ROS2 Humble 自带）解析点云；numpy；无其他第三方。
"""
from __future__ import annotations

import math
import json
import time
from std_msgs.msg import String
from .telemetry_recording import safe_data
import os
import sys
from visualization_msgs.msg import Marker, MarkerArray

_rda_src = os.environ.get('RDA_DWA_SRC', '')
if _rda_src and os.path.isdir(_rda_src) and _rda_src not in sys.path:
    sys.path.insert(0, _rda_src)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

import numpy as np

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

try:
    from sensor_msgs_py import point_cloud2 as _pc2
except Exception:
    _pc2 = None

try:
    from rda_dwa_real_msgs.msg import ObstacleArray, Obstacle
except Exception:
    ObstacleArray = Obstacle = None

from .obstacle_fusion import Clusterer, cluster_bounding_box, infer_type
from .frame_utils import body_to_world_2d, quat_to_yaw, body_pose_to_base_link
from .kf_tracker import GlobalKFTracker
from .localization_input import LocalizationInput
from .perception_core import LidarPerceptionCore, PerceptionParams, point_cloud_xyz


class LidarTrackNode(Node):
    """点云聚类 + 全局系 KF 跟踪 + 稳定 id 发布。"""

    def __init__(self):
        super().__init__('lidar_track_node')
        self._localization_input = LocalizationInput(self)

        # ---------------- 参数 ----------------
        self.declare_parameter('points_topic', '/rslidar_points')
        self.declare_parameter('odom_topic', '/localization')
        self.declare_parameter('obstacles_topic', '/rda/obstacles')
        # 雷达安装高度（base_link → rslidar z 偏移，用于 z 系转换）
        self.declare_parameter('lidar_height', 0.45)
        # === 对齐原项目 cloud_seg 流程（base_link 系）===
        # ① 第一级直通滤波范围（base_link 系）
        self.declare_parameter('pass_x_min', -10.0)
        self.declare_parameter('pass_x_max', 10.0)
        self.declare_parameter('pass_y_min', -5.0)
        self.declare_parameter('pass_y_max', 5.0)
        self.declare_parameter('pass_z_min', -0.5)
        self.declare_parameter('pass_z_max', 0.5)
        # ③ 车体多边形剔除（对齐原项目：agv_length=1.2, agv_width=0.8, outside=0.05）
        self.declare_parameter('vehicle_length', 1.2)
        self.declare_parameter('vehicle_width', 0.8)
        self.declare_parameter('vehicle_outside', 0.05)
        # ⑤ 第二级直通滤波：front 前方区域（base_link 系，聚类用）
        self.declare_parameter('front_x_min', 0.6)
        self.declare_parameter('front_x_max', 3.0)
        # body→base_link x 偏移（与 bridge_node 一致；/localization 给的是 body 位姿）
        self.declare_parameter('body_to_base_link_x', 0.0)
        # 聚类
        self.declare_parameter('cluster_gate', 0.6)
        self.declare_parameter('cluster_min_points', 2)
        self.declare_parameter('cluster_max_radius', 30.0)
        # 卡尔曼 + 关联
        self.declare_parameter('track_dt', 0.1)       # 控制周期（雷达 10Hz 时 0.1）
        self.declare_parameter('track_gate', 1.0)     # 匈牙利关联门限 m
        self.declare_parameter('track_position_slack', 0.15)
        self.declare_parameter('track_lost_ttl', 10)  # 连续 miss 帧数 → 删除（10 帧≈1s）
        self.declare_parameter('kf_q', 0.3)           # 过程噪声
        self.declare_parameter('kf_r', 0.15)          # 观测噪声（聚类质心）
        self.declare_parameter('obstacle_max_speed', 1.5)
        # 动态/静态运动判别（用户场景：动态障碍只有行人，其余车道路障均为静态）
        # 阈值须高于 CV 卡尔曼对静止障碍的稳态速度噪声（实测默认 0.3/0.15，滞回带 0.15~0.3）
        self.declare_parameter('dyn_speed_thr', 0.3)       # KF 速度>此值 → "在动"
        self.declare_parameter('dyn_hold_frames', 5)       # 连续 N 帧在动 → 动态(行人)
        self.declare_parameter('static_speed_thr', 0.15)   # KF 速度<此值 → "静止"
        self.declare_parameter('static_hold_frames', 10)   # 连续 N 帧静止 → 静态
        self.declare_parameter('static_small_thr', 0.6)    # 静态细分：长边≤此值 → block，否则 static_large
        # 定位超时（s）：超过则不刷新障碍（bridge 侧 localization_timeout 会急停）
        self.declare_parameter('loc_timeout', 0.5)
        self.declare_parameter('track_all_around',True)
        self.declare_parameter('voxel_size',.08)
        self.declare_parameter('max_cluster_points',8000)
        self.declare_parameter('max_obstacle_range', 6.0)   # 只检测车周围 5m 内的障碍物
        for name,default in {'ground_filter_enable':True,'ground_min_points':50,
                             'obstacle_height_min':.12,'obstacle_height_max':2.2,
                             'cluster_min_extent':.20,'structural_length':3.5,
                             'structure_tile_size':1.2,'sensor_x':0.,'sensor_y':0.,'sensor_yaw':0.}.items():
            self.declare_parameter(name,default)

        self._points_topic = str(self.get_parameter('points_topic').value)
        self._odom_topic = str(self.get_parameter('odom_topic').value)
        self._obstacles_topic = str(self.get_parameter('obstacles_topic').value)
        # 雷达高度
        self._lidar_h = float(self.get_parameter('lidar_height').value)
        # 第一级直通滤波（base_link 系）
        self._pass_xmin = float(self.get_parameter('pass_x_min').value)
        self._pass_xmax = float(self.get_parameter('pass_x_max').value)
        self._pass_ymin = float(self.get_parameter('pass_y_min').value)
        self._pass_ymax = float(self.get_parameter('pass_y_max').value)
        self._pass_zmin = float(self.get_parameter('pass_z_min').value)
        self._pass_zmax = float(self.get_parameter('pass_z_max').value)
        # 车体多边形参数
        self._veh_len = float(self.get_parameter('vehicle_length').value)
        self._veh_wid = float(self.get_parameter('vehicle_width').value)
        self._veh_out = float(self.get_parameter('vehicle_outside').value)
        self._veh_hl = self._veh_len / 2.0 + self._veh_out
        self._veh_hw = self._veh_wid / 2.0 + self._veh_out
        # front 区域
        self._front_xmin = float(self.get_parameter('front_x_min').value)
        self._front_xmax = float(self.get_parameter('front_x_max').value)
        self._body_to_base_x = float(self.get_parameter('body_to_base_link_x').value)
        self._loc_timeout = float(self.get_parameter('loc_timeout').value)
        self._max_range = float(self.get_parameter('max_obstacle_range').value)

        self._all_around=bool(self.get_parameter('track_all_around').value)
        self._voxel=float(self.get_parameter('voxel_size').value)
        self._max_cluster_points=int(self.get_parameter('max_cluster_points').value)
        self._clusterer = Clusterer(
            gate=float(self.get_parameter('cluster_gate').value),
            min_points=int(self.get_parameter('cluster_min_points').value),
            max_radius=float(self.get_parameter('cluster_max_radius').value),
        )
        self._tracker = GlobalKFTracker(
            dt=float(self.get_parameter('track_dt').value),
            gate=float(self.get_parameter('track_gate').value),
            lost_ttl=int(self.get_parameter('track_lost_ttl').value),
            q=float(self.get_parameter('kf_q').value),
            r=float(self.get_parameter('kf_r').value),
            max_speed=float(self.get_parameter('obstacle_max_speed').value),
            dyn_speed_thr=float(self.get_parameter('dyn_speed_thr').value),
            dyn_hold_frames=int(self.get_parameter('dyn_hold_frames').value),
            static_speed_thr=float(self.get_parameter('static_speed_thr').value),
            static_hold_frames=int(self.get_parameter('static_hold_frames').value),
            static_small_thr=float(self.get_parameter('static_small_thr').value),
            position_slack=float(self.get_parameter('track_position_slack').value),
        )
        self._perception = LidarPerceptionCore(PerceptionParams(
            vehicle_length=float(self.get_parameter('vehicle_length').value),
            vehicle_width=float(self.get_parameter('vehicle_width').value),
            vehicle_outside=float(self.get_parameter('vehicle_outside').value),
            lidar_height=self._lidar_h,max_range=self._max_range,
            x_min=self._pass_xmin if self._all_around else self._front_xmin,
            x_max=self._pass_xmax if self._all_around else self._front_xmax,
            y_min=self._pass_ymin,y_max=self._pass_ymax,z_min=self._pass_zmin,z_max=self._pass_zmax,
            voxel_size=self._voxel,cluster_gate=self._clusterer.gate,
            cluster_min_points=self._clusterer.min_points,max_cluster_points=self._max_cluster_points,
            ground_enabled=bool(self.get_parameter('ground_filter_enable').value),
            ground_min_points=int(self.get_parameter('ground_min_points').value),
            height_min=float(self.get_parameter('obstacle_height_min').value),
            height_max=float(self.get_parameter('obstacle_height_max').value),
            cluster_min_extent=float(self.get_parameter('cluster_min_extent').value),
            structural_length=float(self.get_parameter('structural_length').value),
            structure_tile_size=float(self.get_parameter('structure_tile_size').value),
            sensor_x=float(self.get_parameter('sensor_x').value),sensor_y=float(self.get_parameter('sensor_y').value),
            sensor_yaw=float(self.get_parameter('sensor_yaw').value)))

        # ---------------- 状态 ----------------
        self._loc = None            # (x, y, yaw) 最近定位
        self._loc_t = 0.0           # 最近定位到达时刻（节点时钟）

        # ---------------- 订阅 / 发布 ----------------
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST)
        self._sub_cloud = self.create_subscription(
            PointCloud2, self._points_topic, self._cb_cloud, qos)
        self._sub_odom = self.create_subscription(
            Odometry, self._odom_topic, self._cb_odom, qos)
        if ObstacleArray is not None:
            self._pub_obs = self.create_publisher(
                ObstacleArray, self._obstacles_topic, 10)
        else:
            self._pub_obs = None
            self.get_logger().error(
                'rda_dwa_real_msgs 未找到（未 source car_ros 的 install/setup.bash），'
                '/rda/obstacles 发布被禁用')
        self._pub_perception_diag=self.create_publisher(String,'/rda/perception_diag',10)
        self._perception_seq=0
        self._pub_markers = self.create_publisher(
            MarkerArray, '/rda/obstacles_markers', 10)

        self.get_logger().info(
            f'lidar_track_node 启动: points={self._points_topic} '
            f'odom={self._odom_topic} → {self._obstacles_topic} '
            f'cluster_gate={self._clusterer.gate} track_gate={self._tracker.gate}')

    # ==================================================================
    # 回调
    # ==================================================================
    def _cb_odom(self, msg: Odometry):
        if self._localization_input.uses_tf:
            return
        p = msg.pose.pose
        if self._localization_input.uses_latest_topic:
            self._localization_input.receive_topic(p, msg.header)
            return
        x_body = p.position.x
        y_body = p.position.y
        yaw_body = quat_to_yaw(p.orientation.x, p.orientation.y,
                               p.orientation.z, p.orientation.w)
        # /localization 给的是 body 位姿；补偿成 base_link 位姿后供后续坐标变换使用
        # （rslidar 系 x/y = base_link 系 x/y，故检测点直接和 base_link 位姿对齐）
        x_base, y_base, yaw_base = body_pose_to_base_link(
            x_body, y_body, yaw_body, self._body_to_base_x)
        self._loc = (x_base, y_base, yaw_base)
        self._loc_t = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9

    def _cb_cloud(self,msg:PointCloud2):
        begin=time.monotonic();self._perception_seq+=1
        sample = self._localization_input.sample(self._loc_timeout, now=begin)
        if sample is not None:
            pose, stamp, _ = sample
            self._loc = body_pose_to_base_link(*pose, self._body_to_base_x)
            self._loc_t = stamp
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        self._perception_metrics=dict(schema_version=1,sequence=self._perception_seq,source_stamp=stamp,source_frame=msg.header.frame_id,source_age_s=self.get_clock().now().nanoseconds*1e-9-stamp,raw_points=msg.width*msg.height,raw_bytes=len(msg.data),location_pose_used=self._loc,location_source_stamp=self._loc_t,location_age_s=self.get_clock().now().nanoseconds*1e-9-self._loc_t)
        self._perception_metrics['localization_input'] = self._localization_input.diagnostic()
        if self._localization_input.uses_latest_topic:
            self._perception_metrics.update(
                location_source_stamp=self._localization_input.stamp,
                location_effective_stamp=self._loc_t,
                location_age_s=begin-self._loc_t,
                location_age_clock='monotonic_receive')
        try:self._perception_metrics['reason']=self._process_cloud(msg)
        except Exception as exc:
            self._perception_metrics['reason']='exception';self._perception_metrics['exception']=type(exc).__name__;raise
        finally:
            self._perception_metrics.update(producer_monotonic_ns=time.monotonic_ns(),producer_ros_ns=self.get_clock().now().nanoseconds,duration_s=time.monotonic()-begin,source_dt_s=getattr(self,'_cloud_dt',None))
            self._pub_perception_diag.publish(String(data=json.dumps(safe_data(self._perception_metrics),allow_nan=False)))

    def _process_cloud(self, msg: PointCloud2):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        now=self.get_clock().now().nanoseconds*1e-9
        if stamp<=getattr(self,'_last_cloud_source',-1e9) or now-stamp>.4:return 'cloud_source_stale_or_repeated'
        if msg.point_step<=0 or not msg.data or not {'x','y'}.issubset({f.name for f in msg.fields}):return 'malformed_cloud'
        previous=getattr(self,'_last_cloud_source',None)
        self._cloud_dt=(stamp-previous) if previous is not None else self._tracker.dt
        if self._cloud_dt>.5:self._tracker.reset()
        self._last_cloud_source=stamp;self._cloud_source_stamp=msg.header.stamp
        if self._loc is None:
            return 'no_localization'  # 尚无定位，不发布（bridge 定位超时急停）
        now = (time.monotonic() if self._localization_input.uses_latest_topic
               else self.get_clock().now().nanoseconds * 1e-9)
        if now - self._loc_t > self._loc_timeout:
            return 'localization_stale'  # 定位超时，保持上一帧障碍（bridge obstacle_timeout 兜底清空）
        if self._pub_obs is None:
            return 'obstacle_message_unavailable'

        phase=time.monotonic()
        try:xyz=point_cloud_xyz(msg)
        except (ValueError,TypeError,BufferError):return 'malformed_cloud'
        dets_body,metrics=self._perception.process(xyz,stamp,pose=self._loc)
        self._perception_metrics.update(metrics,filter_cluster_duration_s=time.monotonic()-phase)
        if not metrics['valid']:return metrics['reason']
        if not dets_body:
            self._publish_obstacles(self._tracker.step([],dt=max(.005,min(.5,self._cloud_dt))))
            return 'empty_cloud_predicted_tracks'
        # ⑦ 转到全局系（map），供全局匹配
        rx, ry, rth = self._loc
        dets_world = []
        for (cx, cy, otype, L, W, theta) in dets_body:
            gx, gy, _ = body_to_world_2d(cx, cy, 0.0, rx, ry, rth)
            dets_world.append((gx, gy, otype, L, W, theta+rth))

        self._perception_metrics['detections_world']=dets_world
        # ⑧ 全局系 KF + 匈牙利 → 稳定 id
        phase=time.monotonic();tracks = self._tracker.step(dets_world, dt=max(.005,min(.5,self._cloud_dt)))

        # ⑨ 发布（frame_id='map'）
        self._publish_obstacles(tracks)
        self._perception_metrics.update(track_outputs=tracks,track_count=len(tracks),unknown_count=sum(t[5]=='unknown' for t in tracks),kf_duration_s=time.monotonic()-phase)
        return 'published'

    # ==================================================================
    # 点云解析（对齐原项目 cloud_seg.cpp 流程）
    # ==================================================================
    def _is_point_in_car(self, x: float, y: float) -> bool:
        """射线法判断点是否在车体多边形内（对齐原项目 car_polygon）。

        原项目参数：agv_length=1.2, agv_width=0.8, outside=0.05
        四个角点（base_link 系）：
          ( hl,  hw), ( hl, -hw), (-hl, -hw), (-hl,  hw)
        """
        hl = self._veh_hl
        hw = self._veh_hw
        poly = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
        inside = False
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            if ((poly[i][1] > y) != (poly[j][1] > y)) and \
               (x < (poly[j][0] - poly[i][0]) * (y - poly[i][1]) /
                    (poly[j][1] - poly[i][1]) + poly[i][0]):
                inside = not inside
        return inside

    def _cloud_arround(self, msg: PointCloud2) -> list:
        """对齐原项目 cloud_seg.cpp 的 PointCloudCallback 前半段。

        流程：
          ① 直通滤波 x(-10,10) y(-5,5) z(-0.5,0.5)（base_link 系）
             实现方式：rslidar系 z - lidar_height = base_link系 z（等效于TF到base_link）
          ③ 车体多边形剔除（射线法）
        返回：base_link 系下的 [(x, y)] 周围点云
        """
        if _pc2 is not None:
            try:
                pts = []
                for p in _pc2.read_points(msg, field_names=('x', 'y', 'z'),
                                          skip_nans=True):
                    x = float(p[0])
                    y = float(p[1])
                    z_rs = float(p[2])
                    # z 偏移：rslidar系 → base_link系（雷达安装高度）
                    z_bl = z_rs + self._lidar_h
                    # ① 第一级直通滤波
                    if not (self._pass_xmin <= x <= self._pass_xmax):
                        continue
                    if not (self._pass_ymin <= y <= self._pass_ymax):
                        continue
                    if not (self._pass_zmin <= z_bl <= self._pass_zmax):
                        continue
                    # ③ 车体多边形剔除
                    if self._is_point_in_car(x, y):
                        continue
                    pts.append((x, y))
                return pts
            except Exception:
                pass
        # fallback：手写按 point_step 解析
        out = []
        fmt = getattr(msg, 'fields', [])
        if not fmt:
            return out
        offs = {}
        for f in fmt:
            offs[f.name] = f.offset
        if 'x' not in offs or 'y' not in offs:
            return out
        ps = msg.point_step
        data = msg.data
        zoff = offs.get('z')
        for i in range(0, len(data), ps):
            if i + ps > len(data):
                break
            x = float(struct_unpack_f(data, i + offs['x']))
            y = float(struct_unpack_f(data, i + offs['y']))
            z_bl = 0.0
            if zoff is not None:
                z_rs = float(struct_unpack_f(data, i + zoff))
                z_bl = z_rs + self._lidar_h
            # ① 直通滤波
            if not (self._pass_xmin <= x <= self._pass_xmax):
                continue
            if not (self._pass_ymin <= y <= self._pass_ymax):
                continue
            if zoff is not None and not (self._pass_zmin <= z_bl <= self._pass_zmax):
                continue
            # ③ 车体多边形剔除
            if self._is_point_in_car(x, y):
                continue
            if math.isfinite(x) and math.isfinite(y):
                out.append((x, y))
        return out

    def _cloud_front(self, pts_arround: list) -> list:
        """对齐原项目 cloud_seg.cpp 的 front 滤波。

        从 arround 点云再过滤 x(0.6, 3.0)，得到前方点云（聚类用）。
        """
        pts = []
        for (x, y) in pts_arround:
            if self._front_xmin <= x <= self._front_xmax:
                pts.append((x, y))
        return pts

    # ==================================================================
    # 发布
    # ==================================================================
    def _publish_obstacles(self, tracks):
        """tracks: [(id, x, y, speed, heading, type, L, W)]（全局系）"""
        msg = ObstacleArray()
        msg.header.stamp = getattr(self,'_cloud_source_stamp',self.get_clock().now().to_msg())
        msg.header.frame_id = 'map'
        for row in tracks:
            tid,x,y,speed,heading,otype,L,W=row[:8]
            ob = Obstacle()
            ob.id = int(tid)
            ob.type = otype
            ob.x = float(x)
            ob.y = float(y)
            ob.length = float(L)
            ob.width = float(W)
            ob.heading = float(heading)   # 动态=KF 速度方向(全局系)；静态=0
            ob.speed = float(speed)
            ob.has_shape_heading=len(row)>8
            ob.shape_heading=float(row[8]) if len(row)>8 else float(heading)
            msg.obstacles.append(ob)
        self._pub_obs.publish(msg)
                # ---------- RViz 可视化：障碍物矩形框 + id 文字 ----------
        markers = MarkerArray()
        # 先清掉上一帧所有 marker
        del_all = Marker()
        del_all.header.frame_id = 'map'
        del_all.action = Marker.DELETEALL
        markers.markers.append(del_all)

        for row in tracks:
            tid,x,y,speed,heading,otype,L,W=row[:8]
            # ① 矩形框（CUBE）
            rect = Marker()
            rect.header.frame_id = 'map'
            rect.header.stamp = msg.header.stamp
            rect.ns = 'obstacles_rect'
            rect.id = int(tid)
            rect.type = Marker.CUBE
            rect.action = Marker.ADD
            rect.pose.position.x = float(x)
            rect.pose.position.y = float(y)
            rect.pose.position.z = 0.0
            # 朝向（heading → quaternion）
            rect.pose.orientation.z = math.sin(float(row[8] if len(row)>8 else heading) / 2.0)
            rect.pose.orientation.w = math.cos(float(row[8] if len(row)>8 else heading) / 2.0)
            rect.scale.x = float(L)
            rect.scale.y = float(W)
            rect.scale.z = 0.1          # 薄薄一层，贴地面
            # 颜色：动态障碍红，静态障碍绿
            if 'ped' in otype or 'dyn' in otype:
                rect.color.r = 1.0
                rect.color.g = 0.2
                rect.color.b = 0.2
            else:
                rect.color.r = 0.2
                rect.color.g = 1.0
                rect.color.b = 0.2
            rect.color.a = 0.5
            markers.markers.append(rect)

            # ② id 文字（显示在矩形上方）
            text = Marker()
            text.header.frame_id = 'map'
            text.header.stamp = msg.header.stamp
            text.ns = 'obstacles_text'
            text.id = int(tid)
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(x)
            text.pose.position.y = float(y)
            text.pose.position.z = 0.3   # 框上方 30cm
            text.scale.z = 0.25
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            # text.text = f'id:{tid} {otype}'
            text.text = f'id:{tid} {otype}\n{L:.1f}×{W:.1f}m'

            markers.markers.append(text)

        self._pub_markers.publish(markers)



def struct_unpack_f(data, offset):
    """小端 float32 解析（fallback 用，避免额外依赖）。"""
    import struct
    return struct.unpack_from('<f', data, offset)[0]


def main(args=None):
    rclpy.init(args=args)
    node = LidarTrackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
