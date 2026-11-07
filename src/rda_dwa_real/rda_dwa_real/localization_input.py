"""Use the latest received localization without changing its publishers.

latest_topic caches the most recently received valid pose. Its ROS header is
diagnostic data; a monotonic receipt time drives the disconnect watchdog and
the core update clock. Re-reading the cache cannot renew that receipt time.
The original source-time topic and TF modes remain explicit alternatives.
"""
import math
import time

from .frame_utils import quat_to_yaw


class LocalizationInput:
    def __init__(self, node):
        self.node = node
        node.declare_parameter('localization_source', 'latest_topic')
        node.declare_parameter('localization_parent_frame', 'map')
        node.declare_parameter('localization_child_frame', 'body')
        self.source = str(node.get_parameter('localization_source').value)
        self.parent = str(node.get_parameter('localization_parent_frame').value)
        self.child = str(node.get_parameter('localization_child_frame').value)
        if self.source not in ('topic', 'tf', 'latest_topic'):
            raise ValueError('localization_source must be topic, tf, or latest_topic')
        if not self.parent or not self.child or self.parent == self.child:
            raise ValueError('localization TF needs two different, nonempty frames')
        self.reason = dict(topic='topic', tf='tf_waiting',
                           latest_topic='latest_topic_waiting')[self.source]
        self.stamp = None
        self.age = None
        self._pose = None
        self._received_at = None
        self.receive_age = None
        self.effective_stamp = None
        if self.source == 'tf':
            from rclpy.time import Time
            from tf2_ros import Buffer, TransformListener, TransformException
            self._time = Time
            self._exception = TransformException
            self.buffer = Buffer(node=node)
            self.listener = TransformListener(self.buffer, node)
        node.get_logger().info(
            f'localization_source={self.source} TF={self.parent}->{self.child}')

    @property
    def uses_tf(self):
        return self.source == 'tf'

    @property
    def uses_latest_topic(self):
        return self.source == 'latest_topic'

    def receive_topic(self, pose, header, received_at=None):
        """Cache a valid body pose on message arrival, including old headers."""
        if not self.uses_latest_topic:
            return False
        p, q = pose.position, pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in values) or math.hypot(q.x, q.y, q.z, q.w) < 1e-12:
            self.reason = 'latest_topic_invalid'
            return False
        if header.frame_id not in ('', self.parent):
            self.reason = 'latest_topic_frame_mismatch'
            return False
        now = time.monotonic() if received_at is None else received_at
        if not math.isfinite(now) or (self._received_at is not None and now <= self._received_at):
            return False
        self._pose = (p.x, p.y, quat_to_yaw(q.x, q.y, q.z, q.w))
        self._received_at = now
        self.stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        self.reason = 'latest_topic_valid'
        return True

    def sample(self, max_age, now=None):
        """Return (body pose, effective update stamp, watchdog age), or None.

        latest_topic uses receipt time, allowing delayed, duplicate, and
        irregular ROS headers. TF mode still uses TF's latest common time.
        """
        if self.uses_latest_topic:
            if self._received_at is None:
                self.reason = 'latest_topic_waiting'
                return None
            now = time.monotonic() if now is None else now
            self.receive_age = max(0., now-self._received_at)
            self.age = self.node.get_clock().now().nanoseconds * 1e-9 - self.stamp
            self.effective_stamp = self._received_at
            if self.receive_age > max_age:
                self.reason = 'latest_topic_disconnected'
                return None
            self.reason = 'latest_topic_valid'
            return self._pose, self._received_at, self.receive_age
        if not self.uses_tf:
            return None
        try:
            transform = self.buffer.lookup_transform(
                self.parent, self.child, self._time())
        except self._exception:
            self.reason = 'tf_unavailable'
            return None
        stamp = (transform.header.stamp.sec
                 + transform.header.stamp.nanosec * 1e-9)
        age = self.node.get_clock().now().nanoseconds * 1e-9 - stamp
        self.stamp, self.age = stamp, age
        tr, q = transform.transform.translation, transform.transform.rotation
        values = (tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w, stamp, age)
        if not all(math.isfinite(v) for v in values) or math.hypot(q.x, q.y, q.z, q.w) < 1e-12:
            self.reason = 'tf_invalid'
            return None
        if stamp <= 0 or age > max_age or age < -.05:
            self.reason = 'tf_stale' if age >= 0 else 'tf_future'
            return None
        self.reason = 'tf_valid'
        self.effective_stamp = stamp
        pose = (tr.x, tr.y, quat_to_yaw(q.x, q.y, q.z, q.w))
        return pose, stamp, max(0., age)

    def diagnostic(self):
        return dict(source=self.source, parent_frame=self.parent,
                    child_frame=self.child, reason=self.reason,
                    source_stamp=self.stamp, source_age_s=self.age,
                    receive_age_s=self.receive_age,
                    effective_pose_stamp=self.effective_stamp,
                    pose_stamp_clock='monotonic_receive' if self.uses_latest_topic else 'ros_source')
