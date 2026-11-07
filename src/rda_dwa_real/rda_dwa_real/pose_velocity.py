"""Short robust pose regression; the localization pose itself is unchanged."""
from collections import deque
import math
import numpy as np


class PoseVelocityEstimator:
    def __init__(self, window=.45):
        self.window = window
        self.rows = deque(maxlen=80)

    def reset(self):
        self.rows.clear()

    def update(self, x, y, yaw, stamp):
        if self.rows and stamp <= self.rows[-1][0]:
            return None
        self.rows.append((stamp, x, y, yaw))
        while len(self.rows) > 2 and stamp-self.rows[0][0] > self.window:
            self.rows.popleft()
        if len(self.rows) < 4 or stamp-self.rows[0][0] < .12:
            return 0., 0.
        values = np.asarray(self.rows)
        t = values[:, 0]-stamp
        coords = values[:, 1:].copy()
        coords[:, 2] = np.unwrap(coords[:, 2])
        # Quadratic endpoint derivative follows acceleration without the delay
        # of averaging adjacent differences. Huber reweighting limits outliers.
        design = np.column_stack((np.ones(len(t)), t, t*t))
        weights = np.ones(len(t))
        for _ in range(3):
            root = np.sqrt(weights)
            coef = np.linalg.lstsq(design*root[:, None], coords*root[:, None], rcond=None)[0]
            residual = np.linalg.norm((coords-design@coef)[:, :2], axis=1)
            scale = max(.008, float(np.median(residual))*1.5)
            weights = np.minimum(1., scale/np.maximum(residual, 1e-9))
        vx, vy, omega = coef[1]
        return float(vx*math.cos(yaw)+vy*math.sin(yaw)), float(omega)
