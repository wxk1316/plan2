"""Read the existing localization TF without changing its publishers.

The vehicle's /localization header can refer to an older fusion sample. TF's
map -> odom -> body chain has the current pose and its acquisition timestamp.
All consumers must select the same source; a failed TF lookup never falls back
to the cached topic pose. Core duplicate/jump checks remain in force.
"""
import math

from .frame_utils import quat_to_yaw


class LocalizationInput:
    def __init__(self, node):
        self.node = node
        node.declare_parameter('localization_source', 'topic')
        node.declare_parameter('localization_parent_frame', 'map')
        node.declare_parameter('localization_child_frame', 'body')
        self.source = str(node.get_parameter('localization_source').value)
        self.parent = str(node.get_parameter('localization_parent_frame').value)
        self.child = str(node.get_parameter('localization_child_frame').value)
        if self.source not in ('topic', 'tf'):
            raise ValueError('localization_source must be topic or tf')
        if not self.parent or not self.child or self.parent == self.child:
            raise ValueError('localization TF needs two different, nonempty frames')
        self.reason = 'topic' if self.source == 'topic' else 'tf_waiting'
        self.stamp = None
        self.age = None
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

    def sample(self, max_age):
        """Return (body pose, source stamp, source age), or None.

        Time zero requests TF's latest common time. No blocking timeout is used
        in the control callbacks. Age remains tied to that source time, so
        repeatedly reading the same transform cannot keep the vehicle running.
        """
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
        pose = (tr.x, tr.y, quat_to_yaw(q.x, q.y, q.z, q.w))
        return pose, stamp, max(0., age)

    def diagnostic(self):
        return dict(source=self.source, parent_frame=self.parent,
                    child_frame=self.child, reason=self.reason,
                    source_stamp=self.stamp, source_age_s=self.age)
