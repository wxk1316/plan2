"""Adaptive curvature feedforward preview along the validated rear-axle path.

The LQR error remains at the current projection. This module never constructs
an alternate path or a chord that could cut a collision-checked corner.
"""
import math
import time
import numpy as np
from rda_dwa.motion_safety import minimum_gaps,to_centres

class AdaptivePreview:
    def __init__(self):
        self.distance=0.;self.short_until=-math.inf;self.short_locked=False
        self._cached=None;self._key=None;self._last_update=-math.inf

    @staticmethod
    def _ramp(value,low,high):
        return max(0.,min(1.,(value-low)/max(high-low,1e-6)))

    def choose(self,tracker,s,now,dt):
        p=tracker.p;speed=float(max(0.,min(p.v_max,tracker._v_meas)))
        ahead=float(max(0.,tracker.ref_ahead()))
        if p.adaptive_preview_enabled<.5:
            distance=min(ahead,max(0.,float(tracker._v_meas))*p.preview_time)
            self.distance=distance
            return distance,dict(enabled=False,distance_m=distance,reason='fixed_preview_time')
        key=(tracker._ref_seq,tracker._obstacle_source,tracker._obstacle_at)
        recomputed=False
        if key!=self._key or now-self._last_update>=p.preview_update_period or self._cached is None:
            started=time.perf_counter();recomputed=True
            scan=min(ahead,p.preview_scan_distance)
            positions=np.linspace(s,s+scan,13)
            reference=[tracker._eval_at_s(float(q)) for q in positions]
            curvatures=[r[4] for r in reference]
            peak=max(abs(k) for k in curvatures)
            variation=max(curvatures)-min(curvatures)
            curve_risk=self._ramp(peak,p.preview_curvature_low,p.preview_curvature_high)
            change_risk=self._ramp(variation,p.preview_curvature_change_low,p.preview_curvature_change_high)
            obstacle_fresh=now-tracker._obstacle_at<=p.obstacle_timeout
            gap=math.inf;obstacle_risk=0.;relevant=[]
            if not obstacle_fresh:
                obstacle_risk=1.
            elif tracker._obstacles:
                age=max(0.,now-tracker._obstacle_at)
                relevant=[o for o in tracker._obstacles if math.hypot(o.x-tracker._centre[0],o.y-tracker._centre[1])-tracker.geometry.radius-.5*math.hypot(o.length,o.width)-o.speed*(3.+age)-scan<p.preview_gap_far]
                obstacles=sorted(relevant,key=lambda o:math.hypot(o.x-tracker._centre[0],o.y-tracker._centre[1])-.5*math.hypot(o.length,o.width))[:int(max(1.,p.preview_obstacle_limit))]
                # Accumulate travel time over the emitted speed profile. Bound
                # the risk horizon; latest execution safety is checked elsewhere.
                times=[0.]
                for i in range(1,len(reference)):
                    step=positions[i]-positions[i-1]
                    v=max(.15,.5*(reference[i-1][3]+reference[i][3]))
                    times.append(min(3.,times[-1]+step/v))
                poses=np.asarray([[r[0],r[1],r[2]] for r in reference])[None,:,:]
                gap=float(minimum_gaps(to_centres(poses,tracker.geometry),np.asarray(times),obstacles,tracker.geometry,now-tracker._obstacle_at)[0])
                current=np.asarray([[[tracker._centre[0],tracker._centre[1],tracker._pyaw]]])
                body_gap=float(minimum_gaps(current,np.asarray([0.]),obstacles,tracker.geometry,now-tracker._obstacle_at)[0])
                gap=min(gap,body_gap)
                obstacle_risk=1.-self._ramp(gap,p.preview_gap_near,p.preview_gap_far)
            complex_scene=len(relevant)>p.preview_obstacle_limit
            if complex_scene:obstacle_risk=1.
            risk=max(obstacle_risk,curve_risk,change_risk)
            if not obstacle_fresh:reason='obstacle_input_not_fresh'
            elif complex_scene:reason='complex_scene_short_preview'
            elif obstacle_risk>=max(curve_risk,change_risk) and obstacle_risk>0:reason='obstacle_clearance'
            elif change_risk>=curve_risk and change_risk>0:reason='curvature_change'
            elif curve_risk>0:reason='path_curvature'
            else:reason='open_smooth_path'
            self._cached=dict(risk=risk,reason=reason,predicted_gap_m=gap if math.isfinite(gap) else None,curvature_peak=peak,curvature_variation=variation,obstacle_fresh=obstacle_fresh,obstacle_count=len(tracker._obstacles),risk_relevant_obstacle_count=len(relevant),risk_obstacle_limit=int(p.preview_obstacle_limit),risk_compute_duration_s=time.perf_counter()-started)
            self._key=key;self._last_update=now
        info=dict(self._cached);risk=info['risk']
        if risk>=.75:
            self.short_locked=True;self.short_until=now+p.preview_release_hold
        elif self.short_locked and now>=self.short_until and risk<.55:
            self.short_locked=False
        free=min(p.preview_open_max,speed*p.preview_open_time)
        close=min(p.preview_close_max,speed*p.preview_close_time)
        wanted=close if self.short_locked else free*(1.-risk)+close*risk
        wanted=min(ahead,wanted)
        change=wanted-self.distance
        rate=p.preview_decrease_rate if change<0 else p.preview_increase_rate
        self.distance+=max(-rate*dt,min(rate*dt,change))
        # Near obstacles / sharp curves cap effective preview immediately. The
        # steering angle itself still goes through its rate/acceleration limits.
        cap_applied=self.short_locked and self.distance>close
        if self.short_locked:self.distance=min(self.distance,close)
        self.distance=float(max(0.,min(ahead,free,self.distance)))
        point=tracker._eval_at_s(s+self.distance)
        if self.short_locked and risk<.75:info['risk_reason']=info['reason'];info['reason']='release_hold'
        info.update(enabled=True,risk_recomputed=recomputed,distance_m=self.distance,target_distance_m=wanted,short_locked=self.short_locked,safety_cap_applied=bool(cap_applied),point_rear_xy=[float(point[0]),float(point[1])],point_yaw_rad=float(point[2]),point_curvature=float(point[4]),risk_age_s=now-self._last_update)
        return self.distance,info
