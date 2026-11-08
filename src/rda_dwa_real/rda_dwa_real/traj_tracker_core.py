"""Rear-axle geometric LQR tracking and latched body-centre spin sessions.

References preserve the planner's bounded actuator forecast. Unknown boxes
remain stationary; confirmed moving actors translate at every prediction time.
Execution rechecks fresh observations over its full short horizon. No old tail
or independent x/y/yaw splines alter that geometry.
Localization input is the geometric centre; the conversion stays internal.
"""
import math
from collections import deque
import numpy as np
from .adaptive_preview import AdaptivePreview
from .pose_velocity import PoseVelocityEstimator
from rda_dwa.motion_safety import VehicleGeometry,rollout_rear,to_centres,minimum_gaps,steering_step,speed_step

def wrap_angle(a):return (float(a)+math.pi)%(2*math.pi)-math.pi

def _dlqr(a,b,q,r,iters=200):
    p=np.array(q,dtype=float)
    for _ in range(iters):
        k=np.linalg.solve(r+b.T@p@b,b.T@p@a);pn=q+a.T@p@(a-b@k)
        if np.max(np.abs(pn-p))<1e-12:break
        p=pn
    return k

class TrajTrackerParams:
    def __init__(self):
        self.wheelbase=.69;self.vehicle_length=1.2;self.vehicle_width=.8
        self.max_steer_deg=30.;self.v_max=.8;self.speed_eps=.025
        self.lqr_ds=.1;self.lqr_q_y=8.;self.lqr_q_th=2.;self.lqr_r=4.;self.preview_time=.15
        self.adaptive_preview_enabled=1.0
        self.preview_open_time=.4;self.preview_open_max=.5
        self.preview_close_time=.15;self.preview_close_max=.15
        self.preview_gap_near=.35;self.preview_gap_far=1.5
        self.preview_curvature_low=.2;self.preview_curvature_high=.6
        self.preview_curvature_change_low=.15;self.preview_curvature_change_high=.5
        self.preview_release_hold=.6;self.preview_increase_rate=.3;self.preview_decrease_rate=1.5
        self.preview_update_period=.1;self.preview_scan_distance=.8;self.preview_obstacle_limit=12.0
        self.max_steer_rate_deg=35.;self.max_steer_accel_deg=100.;self.feedback_limit_deg=8.
        self.pid_kp=.3;self.pid_ki=.05;self.pid_kd=0.;self.pid_i_limit=.1;self.pid_overshoot=0.;self.pid_stall_time=1.5
        self.acc_max=.7;self.dec_max=1.2;self.jerk_max=2.;self.v_meas_gain=.3
        self.spin_w_max=.6;self.spin_acc_max=.8;self.rotate_precision_deg=4.;self.rotate_decel_deg=30.
        self.spin_wait_time=3.;self.spin_stop_hold=.5;self.spin_timeout=10.;self.spin_stop_speed=.035
        self.spin_wait_timeout=7.;self.yaw_stop_rate=.06
        self.speed_preview_distance=0.;self.lateral_acc_max=.6;self.stop_window_s=0.
        self.cmd_timeout=.5;self.traj_timeout=.5;self.localization_timeout=.35;self.obstacle_timeout=.4;self.safety_margin=.12
        self.fusion_len=0.;self.keep_behind=0.;self.window_len=3.5;self.tail_join_tol=.35;self.min_pts=2
        self.bspline_ds=.1;self.bspline_min_pts=6;self.ref_window_stale=3.
        # Compatibility parameters are retained for configuration loading only.
        self.lookahead_base=1.2;self.lookahead_short=.6;self.steering_gain=1.;self.max_speed=.8
        self.speed_ramp_max=.2;self.lookahead_v_gain=.75;self.lookahead_max=1.2
        self.traj_len_threshold=.075;self.traj_exit_threshold=.12;self.min_turn_radius=1.195
        self.spin_curvature_sat=.9;self.rotate_entry_deg=10.;self.enter_hold=.3;self.resume_hold=.3
        self.spin_hyst=.2;self.spin_w_deadband=.06;self.spin_min_run=1.;self.recover_delay=3.5
        self.v_resolution=.05;self.omega_resolution=.1

class TrajTrackerCore:
    MODE_ACK='ACK';MODE_SPIN='SPIN';MODE_STOP='STOP';MODE_WAIT_ACK='WAIT_ACK'
    STATE_ACK=0;STATE_HANGUP=1;STATE_SPIN_RUN=2;STATE_SPIN_OK=3;STATE_SPIN_TIMEOUT=4
    def __init__(self,params=None,on_transition=None,on_clear=None):
        self.p=params or TrajTrackerParams();self.geometry=VehicleGeometry(self.p.vehicle_length,self.p.vehicle_width,self.p.wheelbase)
        ds=self.p.lqr_ds;a=np.array([[1.,ds],[0.,1.]]);b=np.array([[0.],[ds/self.p.wheelbase]])
        self._adaptive_preview=AdaptivePreview()
        self._K=_dlqr(a,b,np.diag([self.p.lqr_q_y,self.p.lqr_q_th]),np.array([[self.p.lqr_r]]))
        self._on_transition=on_transition;self._on_clear=on_clear;self.mode=self.MODE_ACK
        self._mode_cmd=0;self._mode_since=0.;self._spin_stage='HOLD';self._spin_target=0.;self._spin_run_since=0.
        self._spin_result=None;self._session=-1;self._consumed_session=-1;self._mode_echo=0;self._mode_echo_at=-1e9
        self._ack_echo_required_since=-1e9;self._ack_requested=False;self._stop_since=None;self._spin_w=0.;self._safety_reason=''
        self._px=self._py=self._pyaw=0.;self._centre=(0.,0.);self._pose_inited=False;self._pose_stamp=None;self._pose_at=-1e9
        self._pose_history=deque(maxlen=100);self._v_meas=0.;self._yaw_rate=0.;self._wheel_max=None;self._wheel_at=-1e9
        self._velocity_estimator=PoseVelocityEstimator()
        self._pose_reinitializations=0
        self._ref=[];self._s_arr=np.zeros(0);self._t_arr=np.zeros(0);self._bs=None;self._ref_seq=-1;self._last_traj_ts=-1e9
        self._s_veh=0.;self._proj_idx=0;self._clear_count=0;self._clear_seq=-1
        self._last_steer=0.;self._steer_rate=0.;self._last_v_out=0.;self._acc=0.;self._pid_i=0.;self._pid_prev_err=0.
        self._pid_act_stall=False;self._diag={};self._transitions=[];self._obstacles=[];self._obstacle_at=-1e9;self._obstacle_source=None
    def set_pose(self,x,y,yaw,dt=0.,stamp=None,arrival=None,allow_reinitialization=False):
        if not all(math.isfinite(float(v)) for v in (x,y,yaw)):self.set_safety('invalid_localization');return
        if stamp is not None and self._pose_stamp is not None and stamp<=self._pose_stamp:return
        rx=x-self.geometry.rear_offset*math.cos(yaw);ry=y-self.geometry.rear_offset*math.sin(yaw)
        source_dt=stamp-self._pose_stamp if stamp is not None and self._pose_stamp is not None else dt
        if self._pose_inited and source_dt>1e-4:
            dx,dy=rx-self._px,ry-self._py;dth=wrap_angle(yaw-self._pyaw)
            if math.hypot(dx,dy)>max(.15,3*source_dt) or abs(dth)>max(.2,4*source_dt):
                if allow_reinitialization and self.mode==self.MODE_ACK and self._last_v_out<.01:
                    self.clear_path('manual_localization_reinitialized');self._pose_history.clear();self._velocity_estimator.reset()
                    self._v_meas=self._yaw_rate=0.;self._pose_reinitializations+=1
                else:self.set_safety('localization_jump');return
        effective_stamp=stamp if stamp is not None else (arrival if arrival is not None else (self._pose_stamp or 0.)+dt)
        estimate=self._velocity_estimator.update(rx,ry,yaw,effective_stamp)
        if estimate is not None:self._v_meas,self._yaw_rate=estimate
        self._px,self._py,self._pyaw=rx,ry,yaw;self._centre=(x,y);self._pose_inited=True;self._pose_stamp=stamp
        if arrival is not None:self._pose_at=arrival
        if stamp is not None:self._pose_history.append((stamp,x,y,yaw))
    def set_mode_ready(self,mode,now):
        mode=int(mode)
        if mode!=self._mode_echo or self._mode_echo_at<self._mode_since:
            self._mode_echo_at=now
        self._mode_echo=mode
    def set_speed_feedback(self,speeds,now):
        vals=list(speeds) if hasattr(speeds,'__iter__') else [float(speeds)]*4
        if len(vals)==4 and all(math.isfinite(float(v)) for v in vals):self._wheel_max=max(abs(v) for v in vals);self._wheel_at=now
    def set_obstacles(self,obstacles,now,stamp=None):
        if stamp is not None and self._obstacle_source is not None and stamp<=self._obstacle_source:return
        self._obstacle_source=stamp;self._obstacles=list(obstacles);self._obstacle_at=now
    def set_safety(self,reason=''):self._safety_reason=str(reason)
    def set_reference(self,pts,now,seq=None):
        if self.mode!=self.MODE_ACK:return
        if seq is not None and seq==self._ref_seq:return
        clean=[]
        for q in pts or []:
            if len(q)<4 or not all(math.isfinite(float(v)) for v in q[:4]):continue
            q=tuple(float(v) for v in q[:6])
            if not all(math.isfinite(v) for v in q):continue
            if clean and math.hypot(q[0]-clean[-1][0],q[1]-clean[-1][1])<1e-5:continue
            clean.append(q)
        if len(clean)<2:self.clear_path('empty_plan');return
        # Replacement is intentional: an old path tail was never validated in
        # the newest obstacle frame and its yaw could create an impossible seam.
        self._ref=[];s=0.
        for i,q in enumerate(clean):
            if i:s+=math.hypot(q[0]-clean[i-1][0],q[1]-clean[i-1][1])
            self._ref.append(dict(s=s,x=q[0],y=q[1],th=q[2],v=max(0.,q[3]),t=now,
                                  target_steer=q[5] if len(q)>5 else None))
        self._s_arr=np.array([r['s'] for r in self._ref]);self._t_arr=np.array([now+r['s']/max(r['v'],.025) for r in self._ref])
        self._last_traj_ts=now;self._ref_seq=seq if seq is not None else self._ref_seq+1
    def clear_path(self,reason=''):
        self._ref=[];self._s_arr=np.zeros(0);self._t_arr=np.zeros(0);self._bs=None;self._s_veh=0.;self._ref_seq=-1;self._clear_count+=1
        if self._on_clear:self._on_clear(reason)
    def consume_clear(self,seq,reason='planner'):
        if seq<=self._clear_seq:return False
        self._clear_seq=seq;self.clear_path(reason);return True
    def _project(self,now=0.):
        best=(math.inf,0.)
        for i,(a,b) in enumerate(zip(self._ref,self._ref[1:])):
            dx,dy=b['x']-a['x'],b['y']-a['y'];l2=dx*dx+dy*dy
            u=max(0.,min(1.,((self._px-a['x'])*dx+(self._py-a['y'])*dy)/max(l2,1e-12)))
            d=(self._px-a['x']-u*dx)**2+(self._py-a['y']-u*dy)**2
            if d<best[0]:best=(d,a['s']+u*(b['s']-a['s']));self._proj_idx=i
        self._s_veh=best[1];return best[1]
    def _eval_at_s(self,s):
        if len(self._ref)<2:return None
        s=max(self._s_arr[0],min(self._s_arr[-1],s));i=min(len(self._ref)-1,max(1,int(np.searchsorted(self._s_arr,s))))
        a,b=self._ref[i-1],self._ref[i];d=max(b['s']-a['s'],1e-6);u=(s-a['s'])/d;th=wrap_angle(b['th']-a['th'])
        return a['x']+u*(b['x']-a['x']),a['y']+u*(b['y']-a['y']),a['th']+u*th,a['v']+u*(b['v']-a['v']),th/d
    def ref_ahead(self):return self._s_arr[-1]-self._s_veh if len(self._s_arr) else 0.
    def _target_steer_at_s(self,s,fallback):
        i=min(len(self._ref)-1,max(1,int(np.searchsorted(self._s_arr,s))))
        a,b=self._ref[i-1],self._ref[i]
        if a['target_steer'] is None or b['target_steer'] is None:return fallback
        u=max(0.,min(1.,(s-a['s'])/max(b['s']-a['s'],1e-6)))
        return a['target_steer']+u*(b['target_steer']-a['target_steer'])
    def _set_mode(self,mode,now):
        if mode!=self.mode:self.mode=mode;self._mode_since=now;self._transitions.append((now,mode));self._on_transition and self._on_transition(mode,now,None)
    def set_traj_mode(self,mode_cmd,now=None,session=None,spin_target=None):
        now=0. if now is None else now;prev=self._mode_cmd;self._mode_cmd=int(mode_cmd)
        if session is None:session=self._consumed_session+1 if mode_cmd==1 and prev!=1 else self._consumed_session
        if mode_cmd==1 and session>self._consumed_session and self.mode==self.MODE_ACK:
            self._session=session;self._consumed_session=session;self._spin_target=self._pyaw if spin_target is None else spin_target
            self._spin_result=None;self._ack_requested=False;self._spin_stage='BRAKE';self._stop_since=None;self._spin_w=0.;self._switch_since=now
            self.clear_path('spin_session');self._set_mode(self.MODE_SPIN,now)
        elif mode_cmd==0 and self.mode in (self.MODE_SPIN,self.MODE_STOP):
            if self._spin_result in (None,'pending'):self._spin_result='aborted'
            self._ack_requested=True
        elif mode_cmd==0 and self.mode==self.MODE_WAIT_ACK:
            self._ack_requested=True
    def bridge_state(self):
        if self.mode in (self.MODE_STOP,self.MODE_WAIT_ACK):
            if self._spin_result=='pending':return 1
            return 3 if self._spin_result=='reached' else 4
        if self.mode==self.MODE_SPIN:return 2 if self._spin_stage=='RUN' else 1
        return 0
    def ack_cmd(self,v,steer):
        self._last_v_out=max(0.,min(self.p.v_max,v));self._last_steer=max(-math.radians(self.p.max_steer_deg),min(math.radians(self.p.max_steer_deg),steer))
        return dict(enable=1,turnmode=0,speed=self._last_v_out,angle=math.degrees(self._last_steer))
    def spin_cmd(self,w):return dict(enable=1,turnmode=4,speed=w,angle=0.)
    def spin_stop_cmd(self):return self.spin_cmd(0.)
    def stop_cmd(self):
        self._last_v_out=0.;self._acc=0.;self._pid_i=0.
        return self.ack_cmd(0.,self._last_steer) # stationary steering is held, never snapped to zero
    def emergency_cmd(self):
        self._last_v_out=0.;self._acc=0.;self._pid_i=0.
        return self.spin_stop_cmd() if self.mode in (self.MODE_SPIN,self.MODE_STOP) else self.stop_cmd()
    def _stopped(self,now):
        wheel_ok=self._wheel_max<self.p.spin_stop_speed if now-self._wheel_at<.4 and self._wheel_max is not None else True
        linear=abs(self._v_meas);angular=abs(self._yaw_rate)
        if self.p.stop_window_s>0 and self._pose_history:
            latest=self._pose_history[-1][0]
            rows=[r for r in self._pose_history if r[0]>=latest-self.p.stop_window_s]
            if len(rows)>=6 and rows[-1][0]-rows[0][0]>=max(.25,self.p.stop_window_s*.8):
                a=np.asarray(rows);tt=a[:,0]-a[:,0].mean();den=max(float(tt@tt),1e-9)
                linear=math.hypot(float(tt@a[:,1]/den),float(tt@a[:,2]/den))
                angular=abs(float(tt@np.unwrap(a[:,3])/den))
        return linear<self.p.spin_stop_speed and angular<self.p.yaw_stop_rate and wheel_ok
    def _steer(self,target,dt):
        max_rate=math.radians(self.p.max_steer_rate_deg);max_acc=math.radians(self.p.max_steer_accel_deg)
        delta,rate=steering_step(self._last_steer,self._steer_rate,target,dt,max_rate,max_acc)
        self._steer_rate=float(rate)
        return float(delta)
    def step(self,now,dt,spin_target=None):
        dt=max(.001,min(.1,dt))
        if self._safety_reason:
            if self.mode==self.MODE_SPIN:
                self._ack_echo_required_since=-1e9 if self._spin_stage=='BRAKE' else now
                self._spin_result='aborted'
                self._set_mode(self.MODE_WAIT_ACK if self._spin_stage=='BRAKE' else self.MODE_STOP,now)
            self._diag={'branch':'SAFETY_STOP','reason':self._safety_reason}
            if self.mode in (self.MODE_STOP,self.MODE_WAIT_ACK):return self._step_spin(now,dt)
            return self.emergency_cmd()
        if self.mode in (self.MODE_SPIN,self.MODE_STOP,self.MODE_WAIT_ACK):return self._step_spin(now,dt)
        if len(self._ref)<2:self._diag={'branch':'NO_REF'};return self.stop_cmd()
        s=self._project();ref=self._eval_at_s(s);xr,yr,thr,vr,kr=ref
        ey=-(self._px-xr)*math.sin(thr)+(self._py-yr)*math.cos(thr);eth=wrap_angle(self._pyaw-thr)
        preview_distance,preview_diag=self._adaptive_preview.choose(self,s,now,dt)
        look=self._eval_at_s(s+preview_distance)
        # Curvature for feedforward can look ahead; error NEVER advances by wall time.
        ff=self._target_steer_at_s(s,math.atan(self.p.wheelbase*look[4]))
        feedback_y=math.copysign(max(0.,abs(ey)-.015),ey)
        feedback_th=math.copysign(max(0.,abs(eth)-math.radians(.7)),eth)
        corr=-(self._K@np.array([feedback_y,feedback_th])).item()
        corr=max(-math.radians(self.p.feedback_limit_deg),min(math.radians(self.p.feedback_limit_deg),corr))
        target=max(-math.radians(self.p.max_steer_deg),min(math.radians(self.p.max_steer_deg),ff+corr))
        delta=self._steer(target,dt)
        # Reference speed is the desired profile; global cap and the final
        # collision guard constrain lagged commands during a smooth deceleration.
        velocity_ref=self._eval_at_s(s+self.p.speed_preview_distance)[3]
        cap=min(velocity_ref,self.p.v_max,math.sqrt(self.p.lateral_acc_max/max(abs(math.tan(delta)/self.p.wheelbase),.01)))
        cap*=max(.35,math.cos(min(abs(eth),math.pi/2)))
        err=cap-self._v_meas;self._pid_i=max(-self.p.pid_i_limit,min(self.p.pid_i_limit,self._pid_i+err*dt))
        wanted=min(cap,max(0.,cap+self.p.pid_kp*err+self.p.pid_ki*self._pid_i))
        v,acceleration=speed_step(self._last_v_out,self._acc,wanted,dt,self.p.acc_max,self.p.dec_max,self.p.jerk_max)
        self._acc=float(acceleration);v=min(self.p.v_max,float(v))
        execution_gap=None
        if now-self._obstacle_at<=self.p.obstacle_timeout:
            rear,_,tt=rollout_rear(self._px,self._py,self._pyaw,max(0.,self._v_meas),self._last_steer,[v],[delta],self.geometry,max(.8,max(self._v_meas,0.)/self.p.dec_max+.6),.05,dec=self.p.dec_max,steer_rate=math.radians(self.p.max_steer_rate_deg),jerk=self.p.jerk_max,steer_accel=math.radians(self.p.max_steer_accel_deg),initial_acc=self._acc,initial_steer_rate=self._steer_rate,delay=.1)
            bound=(max(v,self._v_meas,0.)+abs(v*math.tan(delta)/self.p.wheelbase)*self.geometry.radius)*.025
            execution_gap=float(minimum_gaps(to_centres(rear,self.geometry),tt,self._obstacles,self.geometry,now-self._obstacle_at,bound)[0])
            if execution_gap<self.p.safety_margin:
                self._diag={'branch':'EXECUTION_GUARD','min_predicted_gap':execution_gap,'required_margin':self.p.safety_margin,'preview':preview_diag,'obstacle_age_s':now-self._obstacle_at};return self.stop_cmd()
        self._diag={'branch':'LQR','preview':preview_diag,'speed_preview_distance_m':self.p.speed_preview_distance,'min_predicted_gap':execution_gap,'required_margin':self.p.safety_margin,'e_y':ey,'e_th':eth,'kappa_ref':look[4],'delta_ff':ff,'correction':corr,'v_ref':cap,'v_out':v,'s':s,'ahead':self.ref_ahead()}
        return self.ack_cmd(v,delta)
    def _step_spin(self,now,dt):
        if self.mode==self.MODE_WAIT_ACK:
            if self._ack_requested and self._stopped(now) and self._mode_echo==0 and self._mode_echo_at>=self._ack_echo_required_since and now-self._mode_since>=self.p.spin_wait_time:
                self._set_mode(self.MODE_ACK,now);self._spin_result=None;self._last_steer=0.;self._steer_rate=0.;self.clear_path('ack_ready')
            self._diag={'branch':'ACK_MODE_WAIT'};return self.stop_cmd()
        if self.mode==self.MODE_STOP:
            if self._stopped(now):self._stop_since=now if self._stop_since is None else self._stop_since
            else:self._stop_since=None
            held=self._stop_since is not None and now-self._stop_since>=self.p.spin_stop_hold
            if self._spin_result=='pending' and held:
                err=wrap_angle(self._spin_target-self._pyaw)
                if abs(err)<=math.radians(self.p.rotate_precision_deg):self._spin_result='reached'
                elif now-self._spin_run_since<self.p.spin_timeout:
                    # Coasting moved outside tolerance: correct in the SAME
                    # session, with the ORIGINAL execution timeout budget.
                    self._spin_stage='RUN';self._spin_w=0.;self._set_mode(self.MODE_SPIN,now)
                else:self._spin_result='timeout'
            if self._ack_requested and held and self._spin_result in ('reached','timeout','aborted'):
                self._ack_echo_required_since=now
                self._set_mode(self.MODE_WAIT_ACK,now)
            self._diag={'branch':'SPIN_RESULT_HOLD','spin_err':wrap_angle(self._spin_target-self._pyaw)};return self.spin_stop_cmd()
        err=wrap_angle(self._spin_target-self._pyaw)
        if self._ack_requested:
            self._ack_echo_required_since=-1e9 if self._spin_stage=='BRAKE' else now
            self._spin_result='aborted';self._set_mode(self.MODE_WAIT_ACK if self._spin_stage=='BRAKE' else self.MODE_STOP,now)
            return self.stop_cmd() if self.mode==self.MODE_WAIT_ACK else self.spin_stop_cmd()
        if self._spin_stage=='BRAKE':
            stopped=self._stopped(now)
            self._stop_since=(now if self._stop_since is None else self._stop_since) if stopped else None
            if self._stop_since is not None and now-self._stop_since>=.3:
                self._spin_stage='HOLD';self._switch_since=now
            elif now-self._mode_since>self.p.spin_wait_timeout:
                self._spin_result='timeout';self._ack_echo_required_since=-1e9
                self._set_mode(self.MODE_WAIT_ACK,now)
            self._diag={'branch':'SPIN_BRAKE','spin_err':err}
            return self.stop_cmd() # ACK wheels remain in place until physical stop
        if self._spin_stage=='HOLD':
            stopped=self._stopped(now)
            self._stop_since=(now if self._stop_since is None else self._stop_since) if stopped else None
            echo=self._mode_echo==4 and self._mode_echo_at>=self._switch_since
            if echo and self._stop_since is not None and now-self._stop_since>=.3 and now-max(self._switch_since,self._mode_echo_at)>=self.p.spin_wait_time:
                self._spin_stage='RUN';self._spin_run_since=now
            elif now-self._mode_since>self.p.spin_wait_timeout:
                self._spin_result='timeout';self._set_mode(self.MODE_STOP,now)
            self._diag={'branch':'SPIN_HOLD','spin_err':err};return self.spin_stop_cmd()
        if now-self._spin_run_since>self.p.spin_timeout:
            self._spin_result='timeout';self._set_mode(self.MODE_STOP,now);return self.spin_stop_cmd()
        if abs(err)<=math.radians(self.p.rotate_precision_deg):
            self._spin_result='pending';self._set_mode(self.MODE_STOP,now);self._diag={'branch':'SPIN_TARGET_BRAKE','spin_err':err};return self.spin_stop_cmd()
        wanted=math.copysign(min(self.p.spin_w_max,self.p.spin_w_max*abs(err)/math.radians(self.p.rotate_decel_deg)),err)
        self._spin_w+=max(-self.p.spin_acc_max*dt,min(self.p.spin_acc_max*dt,wanted-self._spin_w))
        self._diag={'branch':'SPIN_RUN','spin_err':err};return self.spin_cmd(self._spin_w)
    def status_line(self,extra=''):
        d=self._diag
        return f"mode={self.mode} branch={d.get('branch','-')} v={self._last_v_out:.2f} steer={math.degrees(self._last_steer):.1f} v_ref={d.get('v_ref',0):.2f} e_y={d.get('e_y',0):+.3f} e_th={math.degrees(d.get('e_th',0)):+.1f} k_ref={d.get('kappa_ref',0):+.3f} preview={d.get('preview',{}).get('distance_m',0):.3f} s={self._s_veh:.2f} ahead={self.ref_ahead():.2f} n={len(self._ref)} spin_err={math.degrees(d.get('spin_err',0)):+.1f} clear_seq={self._clear_seq} seq={self._ref_seq} session={self._session} reason={d.get('reason','')} {extra}"
    @property
    def diag(self):return dict(self._diag)
