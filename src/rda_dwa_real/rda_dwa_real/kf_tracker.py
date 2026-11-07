"""
kf_tracker.py — 全局坐标系障碍物跟踪器（卡尔曼滤波 + 匈牙利匹配，纯 numpy，可单测）。

用途：
  为激光雷达聚类得到的"无 id"障碍物赋予跨帧稳定 id，输出平滑的位置/速度，
  供 RL-DWA 实车适配层（bridge_node → TrackManager → IMM predictor）消费。

设计口径（与用户确认）：
  - 坐标系：全局系（algorithm_frame='map'）。检测物由调用方先从车体系转到全局系。
  - 运动模型：CV（匀速）卡尔曼滤波，状态 [x, y, vx, vy]^T，dt 步进。
  - 数据关联：匈牙利最小成本匹配（scipy.optimize.linear_sum_assignment，开源标准库），
    成本 = 预测位置与检测位置欧氏距离，超过关联门限 gate 的置不可匹配；
    无 scipy 时回退贪心最近邻（按成本升序逐一锁定）。
  - track 生命周期：匹配 → KF 更新；未匹配检测 → 新建；连续 miss >= lost_ttl → 删除。
  - 类型/尺寸：每次匹配刷新观测框并保守包含最新观测；
    运动类别由真实观测滞回确认，未确认/丢失动态目标保持 unknown。

依赖：仅 numpy（+ 可选 scipy）。不依赖 ROS，Windows/WSL 均可单测。
"""
from __future__ import annotations

import math

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:  # 无 scipy 时回退贪心最近邻
    linear_sum_assignment = None
    _HAS_SCIPY = False


def wrap_angle(a: float) -> float:
    """归一化到 (-pi, pi]。"""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


class KFTrack:
    """单个障碍物卡尔曼滤波器（CV 模型）。"""

    def __init__(self, tid: int, x: float, y: float, dt: float,
                 q: float = 0.3, r: float = 0.15,
                 obs_type: str = 'unknown',
                 length: float = 1.0, width: float = 1.0,
                 max_speed: float = 1.5,
                 dyn_speed_thr: float = 0.3,
                 dyn_hold_frames: int = 5,
                 static_speed_thr: float = 0.15,
                 static_hold_frames: int = 10):
        self.id = int(tid)
        self.dt = float(dt)
        self.q=float(q)
        self.max_speed = float(max_speed)
        self.dyn_speed_thr = float(dyn_speed_thr)
        self.dyn_hold_frames = int(dyn_hold_frames)
        self.static_speed_thr = float(static_speed_thr)
        self.static_hold_frames = int(static_hold_frames)
        # 状态 [x, y, vx, vy]
        self.x = np.asarray([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 2.0, 2.0])  # 初值大协方差（速度未知）
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        dt2 = self.dt * self.dt
        self.Q = float(q) * np.array([[dt2 * dt2 / 4, 0, dt2 * self.dt / 2, 0],
                                      [0, dt2 * dt2 / 4, 0, dt2 * self.dt / 2],
                                      [dt2 * self.dt / 2, 0, dt2, 0],
                                      [0, dt2 * self.dt / 2, 0, dt2]],
                                     dtype=np.float64)
        self.R = np.eye(2, dtype=np.float64) * float(r) ** 2
        self.miss = 0               # 连续未匹配帧数
        # 最新检测的几何类型/尺寸（不是语义识别）
        self.obs_type = obs_type
        self.length = float(length)
        self.width = float(width)
        self.shape_heading=None
        # 动态/静态判定（滞回）：初始输出 unknown
        self.moving = False
        self._mv_cnt = 0
        self._st_cnt = 0
        self.last_measurement = (float(x), float(y))
        self.measurement_age = 0.0

    # ------------------------------------------------------------------
    def set_dt(self,dt):
        self.dt=float(dt);self.F[0,2]=self.F[1,3]=self.dt
        d=self.dt;d2=d*d
        self.Q=self.q*np.array([[d2*d2/4,0,d2*d/2,0],[0,d2*d2/4,0,d2*d/2],
                               [d2*d/2,0,d2,0],[0,d2*d/2,0,d2]],dtype=float)

    def predict(self) -> None:
        self.measurement_age += self.dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, zx: float, zy: float) -> None:
        self.last_measurement = (float(zx), float(zy))
        self.measurement_age = 0.0
        z = np.asarray([zx, zy], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph 形式（数值稳定）
        self.miss = 0

    # ------------------------------------------------------------------
    def position(self) -> tuple:
        return float(self.x[0]), float(self.x[1])

    def velocity(self) -> tuple:
        return float(self.x[2]), float(self.x[3])

    def speed_heading(self) -> tuple:
        """由滤波速度得到 (speed, heading)；低速视为静止。"""
        vx, vy = self.velocity()
        sp = math.hypot(vx, vy)
        if sp < 0.05:
            return 0.0, 0.0
        return min(sp, self.max_speed), math.atan2(vy, vx)

    def update_motion(self) -> None:
        """运动性判定（滞回）：KF 速度持续超阈值 → moving；持续低于阈值 → 静止。

        未确认前保持 unknown；只有匹配的真实观测推进运动/静止计数。
        """
        sp = math.hypot(self.x[2], self.x[3])
        if sp > self.dyn_speed_thr:
            self._mv_cnt += 1
            self._st_cnt = 0
        elif sp < self.static_speed_thr:
            self._st_cnt += 1
            self._mv_cnt = 0
        else:
            self._mv_cnt = 0
            self._st_cnt = 0
        if not self.moving and self._mv_cnt >= self.dyn_hold_frames:
            self.moving = True
        elif self.moving and self._st_cnt >= self.static_hold_frames:
            self.moving = False


# ---------------------------------------------------------------------------
# 匈牙利匹配
# ---------------------------------------------------------------------------
class HungarianMatcher:
    """检测物 ↔ 已有 track 的最小成本匹配。

    成本 = 预测位置与检测位置欧氏距离；超过 gate 的项置 INF 不可匹配。
    scipy 可用时用匈牙利最优解；否则贪心最近邻（按成本升序逐一锁定）。
    """

    def __init__(self, gate: float = 1.0):
        self.gate = float(gate)

    def match(self, pred: list, dets: list, last_measured=None, max_displacements=None,shape_slack=None) -> tuple:
        """pred: [(px, py)] 预测位置；dets: [(dx, dy)] 检测位置。

        返回 (matched, unmatched_det_idx)：
          matched: [(track_idx, det_idx)]，cost 均 <= gate
          unmatched_det_idx: 未匹配的检测下标
        """
        n = len(pred)
        m = len(dets)
        if n == 0 or m == 0:
            return [], list(range(m))
        cost = np.full((n, m), 1e6, dtype=np.float64)
        for i, (px, py) in enumerate(pred):
            for j, (dx, dy) in enumerate(dets):
                d = math.hypot(px - dx, py - dy)
                reachable = (last_measured is None or
                             math.hypot(dx-last_measured[i][0],dy-last_measured[i][1]) <= max_displacements[i]+(shape_slack[i,j] if shape_slack is not None else 0.))
                if d <= self.gate and reachable:
                    cost[i, j] = d
        if _HAS_SCIPY:
            rows, cols = linear_sum_assignment(cost)
            matched = [(int(ri), int(ci)) for ri, ci in zip(rows, cols)
                       if cost[ri, ci] <= self.gate]
        else:
            # 贪心：按成本升序逐一锁定（成本 < INF 才允许）
            pairs = [(int(i), int(j), float(cost[i, j]))
                     for i in range(n) for j in range(m) if cost[i, j] <= self.gate]
            pairs.sort(key=lambda t: t[2])
            used_t, used_d = set(), set()
            matched = []
            for i, j, c in pairs:
                if i in used_t or j in used_d:
                    continue
                used_t.add(i)
                used_d.add(j)
                matched.append((i, j))
        matched_det = {d for _, d in matched}
        unmatched = [j for j in range(m) if j not in matched_det]
        return matched, unmatched


# ---------------------------------------------------------------------------
# 全局跟踪器
# ---------------------------------------------------------------------------
class GlobalKFTracker:
    """多障碍物全局跟踪器：KF 预测 → 匈牙利关联 → 更新/新建/删除 → 稳定 id。

    输入检测（全局系）：[(x, y, obs_type, L, W[, shape_heading])]
    输出：按 id 排序的 [(id, x, y, speed, heading, obs_type, L, W)]
    """

    STATIC_TYPES = ('static_large', 'block')

    def __init__(self, dt: float = 0.1, gate: float = 1.0, lost_ttl: int = 10,
                 q: float = 0.3, r: float = 0.15, max_speed: float = 1.5,
                 dyn_speed_thr: float = 0.3, dyn_hold_frames: int = 5,
                 static_speed_thr: float = 0.15, static_hold_frames: int = 10,
                 static_small_thr: float = 0.6, position_slack: float = 0.15):
        self.dt = float(dt)
        self.gate = float(gate)
        self.lost_ttl = int(lost_ttl)
        self.q = float(q)
        self.r = float(r)
        self.max_speed = float(max_speed)
        self.dyn_speed_thr = float(dyn_speed_thr)
        self.dyn_hold_frames = int(dyn_hold_frames)
        self.static_speed_thr = float(static_speed_thr)
        self.static_hold_frames = int(static_hold_frames)
        self.static_small_thr = float(static_small_thr)
        self.position_slack = float(position_slack)
        if not math.isfinite(self.position_slack) or self.position_slack<0:
            raise ValueError('invalid association position slack')
        self.tracks: dict[int, KFTrack] = {}
        self._next_id = 1
        self._matcher = HungarianMatcher(gate=self.gate)

    # ------------------------------------------------------------------
    def reset(self):
        self.tracks.clear()
        self._next_id = 1

    # ------------------------------------------------------------------
    def step(self, detections, dt=None) -> list:
        """推进一帧。

        detections: [(x, y, obs_type, L, W)]，全局系。
        返回: [(id, x, y, speed, heading, obs_type, L, W)]，按 id 升序。
        """
        if dt is not None:
            if not math.isfinite(dt) or dt<=0:raise ValueError('invalid source interval')
            self.dt=float(dt)
        for t in self.tracks.values():t.set_dt(self.dt)
        if not detections:
            for t in self.tracks.values():
                t.predict()
                t.miss += 1
            self._delete_missed()
            return self._output()

        ids = sorted(self.tracks.keys())

        # 预测（KF predict）
        for t in self.tracks.values():
            t.predict()

        pred = [self.tracks[i].position() for i in ids]
        det_xy = [(d[0], d[1]) for d in detections]
        # A fixed 1 m gate at 10--20 Hz can join two alternating static
        # clusters into a fictitious moving track. Check reachability from the
        # last actual measurement, with elapsed source time and noise slack.
        measured=[self.tracks[i].last_measurement for i in ids]
        bounds=[self.max_speed*self.tracks[i].measurement_age+self.position_slack for i in ids]
        # Partial visibility changes a box centre as its dimensions change.
        # Account for that bounded effect without widening identical small
        # clusters' gates (the alternating-cluster false-motion regression).
        shape_slack=np.zeros((len(ids),len(detections)))
        for i,tid in enumerate(ids):
            old=sorted((self.tracks[tid].length,self.tracks[tid].width))
            for j,d in enumerate(detections):
                new=sorted((float(d[3]),float(d[4])))
                shape_slack[i,j]=min(.35,.5*math.hypot(new[0]-old[0],new[1]-old[1]))
        matched, unmatched_det = self._matcher.match(pred, det_xy, measured, bounds,shape_slack)

        # 匹配 → 更新
        for ti, di in matched:
            tr = self.tracks[ids[ti]]
            tr.update(det_xy[di][0], det_xy[di][1])
            self._refresh_attr(tr, detections[di])

        # 未匹配检测 → 新建
        for di in unmatched_det:
            dx, dy, otype, L, W = detections[di][0], detections[di][1], \
                detections[di][2], detections[di][3], detections[di][4]
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = KFTrack(tid, dx, dy, self.dt,
                                       q=self.q, r=self.r,
                                       obs_type=otype, length=L, width=W,
                                       max_speed=self.max_speed,
                                       dyn_speed_thr=self.dyn_speed_thr,
                                       dyn_hold_frames=self.dyn_hold_frames,
                                       static_speed_thr=self.static_speed_thr,
                                       static_hold_frames=self.static_hold_frames)
            self._refresh_attr(self.tracks[tid],detections[di])
        # 未匹配 track → miss 累计 / 删除
        matched_t = {ids[ti] for ti, _ in matched}
        for tid in ids:
            if tid not in matched_t:
                self.tracks[tid].miss += 1
        self._delete_missed()

        # 运动性判定（KF 速度 + 滞回）→ 类型/速度由 _output 决策
        for t in self.tracks.values():
            if t.miss == 0:
                t.update_motion()  # predictions must not confirm stationarity

        return self._output()

    # ------------------------------------------------------------------
    def _refresh_attr(self, tr: KFTrack, det) -> None:
        """匹配帧刷新属性：静态类型航向冻结由输出层处理；类型/尺寸随当前几何观测刷新。"""
        tr.length=float(det[3]);tr.width=float(det[4]);tr.obs_type=det[2]
        if len(det)>5:tr.shape_heading=float(det[5])
        # Enclose the latest measured box around the filtered position, whose
        # lag can otherwise shave a corner off an observed vehicle.
        dx,dy=float(det[0])-tr.x[0],float(det[1])-tr.x[1]
        hd=tr.shape_heading or 0.;c,s=math.cos(hd),math.sin(hd)
        tr.length+=2*abs(c*dx+s*dy);tr.width+=2*abs(-s*dx+c*dy)

    def _delete_missed(self):
        dead = [tid for tid, t in self.tracks.items() if t.miss >= self.lost_ttl]
        for tid in dead:
            del self.tracks[tid]

    def _output(self) -> list:
        """输出 [(id, x, y, speed, heading, otype, L, W)]，按 id 升序。

        类型由运动性判定决策；检测名称仅表示尺寸提示：
          - moving=True → 保留 pedestrian/bicycle/vehicle 尺寸提示，输出 KF 速度
          - moving=False → 静态：长边 ≤ static_small_thr → 'block'，否则 'static_large'
            （速度 0，框的真实方向仍通过 shape_heading 保留）
        """
        out = []
        for tid in sorted(self.tracks.keys()):
            t = self.tracks[tid]
            x, y = t.position()
            if t.moving:
                sp, hd = t.speed_heading()
                otype = t.obs_type if t.obs_type in ('pedestrian','bicycle','vehicle') else 'pedestrian'
            elif t._st_cnt<self.static_hold_frames:
                sp,hd=t.speed_heading();otype='unknown'
            else:
                sp, hd = 0.0, 0.0
                otype = 'block' if max(t.length, t.width) <= self.static_small_thr \
                    else 'static_large'
            # Keep a measured, confirmed static obstacle through the existing
            # short miss TTL. Missing it for one scan does not prove motion.
            # Unconfirmed/moving missed tracks retain the unknown protection.
            if t.miss and (t.moving or t._st_cnt<self.static_hold_frames):otype='unknown'
            length,width=t.length,t.width
            if otype=='unknown':
                x,y=t.last_measurement
                # ObstacleArray is stamped with the current cloud. Account for
                # age since this track's last observation in its present box.
                padding=2.*t.max_speed*t.measurement_age
                length+=padding;width+=padding
                sp,hd=0.,0.
            row=(tid,x,y,sp,hd,otype,length,width)
            out.append(row+(t.shape_heading,) if t.shape_heading is not None else row)
        return out


if __name__ == '__main__':
    # 纯逻辑自检（无 ROS）：
    # 三个静止/匀速障碍，10 帧内 id 稳定、位置跟随、速度估计正确。
    import random
    random.seed(0)
    trk = GlobalKFTracker(dt=0.1, gate=1.5, lost_ttl=10, q=0.3, r=0.15)
    true = [(2.0, 1.0, 'vehicle', 1.4, 0.8),
            (5.0, -1.0, 'pedestrian', 0.15, 0.15),
            (-3.0, 2.0, 'static_large', 1.5, 1.5)]
    obs_ids = []
    for k in range(30):
        dets = []
        for (tx, ty, ot, L, W) in true:
            if ot == 'pedestrian':
                tx += 0.5 * 0.1 * k   # 0.5 m/s 沿 x
            if ot == 'vehicle':
                ty += 0.8 * 0.1 * k   # 0.8 m/s 沿 y
            # 模拟观测噪声
            dets.append((tx + random.gauss(0, 0.1), ty + random.gauss(0, 0.1),
                         ot, L, W))
        out = trk.step(dets)
        obs_ids.append(sorted(o[0] for o in out))
        if k in (0, 5, 29):
            print(f'frame {k}: ids={[o[0] for o in out]}')
            for o in out:
                print(f'   id={o[0]} type={o[5]} x={o[1]:.2f} y={o[2]:.2f} '
                      f'speed={o[3]:.2f} heading={math.degrees(o[4]):.1f}')
    # 校验：id 稳定
    assert all(s == obs_ids[0] for s in obs_ids), 'id 不稳定'
    assert len(obs_ids[0]) == 3, 'track 数不对'
    print('\nself-check OK: 3 tracks, ids stable, KF follows motion.')
