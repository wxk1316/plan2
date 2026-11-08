"""Arrival/source-clock safe supervisor. Localization input is body centre.

Planner/path coordinates are rear axle. Spin footprint is body centre. Internal
reference conversion never modifies localization publications or TF.
"""
import math,time
import numpy as np
from .pose_velocity import PoseVelocityEstimator
from .vehicle_bridge import VehicleBridgeCore
from rda_dwa.motion_safety import VehicleGeometry,TrackedObstacle,rollout_rear,to_centres,minimum_gaps,spin_is_safe

class RealtimeVehicleBridgeCore(VehicleBridgeCore):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # The ROS timer publishes diagnostics before the first goal callback.
        # This supervisor overrides the legacy step that initialized _goal.
        self._goal=None
        self.geometry=VehicleGeometry(self.ack_model.p.length,self.ack_model.p.width,self.ack_model.p.wheelbase)
        self._pose_source=None;self._pose_prior=None;self._pose_fault=False;self._v_meas=0.;self._mode_ready=0
        self._mode_arrival=-1e9;self._session=0;self._session_target=0.;self._blocked_since=None
        self._episode_anchor=None;self._episode_retries=0;self._cooldown=-1e9;self._request_since=0.
        self._last_received_obs=[];self._obs_source=None;self._tracker_angle=0.;self._tracker_mode=0
        self._t_loc=self._t_obs=-1e9
        self._goal_requests=0;self._recovery_goal=None;self._recovery_failed=False
        self._recovery_elapsed=0.;self._last_step=None;self._stall_elapsed=0.
        self._motion_anchor=None;self._motion_at=0.;self._standstill_elapsed=0.;self._reconnect=None
        self._velocity_estimator=PoseVelocityEstimator()
        self._tracker_acc=0.;self._tracker_steer_rate=0.;self._tracker_cmd_at=None
    def centre(self):
        return (self.robot.x+self.geometry.rear_offset*math.cos(self.robot.theta),self.robot.y+self.geometry.rear_offset*math.sin(self.robot.theta))
    def set_localization(self,x,y,theta,stamp=None,arrival=None):
        now=time.monotonic() if arrival is None else arrival
        if not all(math.isfinite(float(v)) for v in (x,y,theta)):
            self._pose_fault=True;return
        if stamp is not None and self._pose_source is not None and stamp<=self._pose_source:return
        rx=x-self.geometry.rear_offset*math.cos(theta);ry=y-self.geometry.rear_offset*math.sin(theta)
        if self._pose_prior is not None and stamp is not None and self._pose_source is not None:
            dt=stamp-self._pose_source
            if dt<=0:return
            dx=rx-self._pose_prior[0];dy=ry-self._pose_prior[1];dyaw=(theta-self._pose_prior[2]+math.pi)%(2*math.pi)-math.pi
            if self._goal_received and (math.hypot(dx,dy)>max(.15,3.*dt) or abs(dyaw)>max(.2,4.*dt)):
                self._pose_fault=True;return
        estimate=self._velocity_estimator.update(rx,ry,theta,stamp if stamp is not None else now)
        if estimate is not None:self._v_meas,self._omega_meas=estimate
        self._pose_source=stamp;self._pose_prior=(rx,ry,theta);self._t_loc=now
        self.robot.x,self.robot.y,self.robot.theta=rx,ry,theta
        self.state.x,self.state.y,self.state.theta=rx,ry,theta
    def set_goal(self,gx,gy,stamp=None,frame='map'):
        if not all(math.isfinite(float(v)) for v in (gx,gy)):return
        if frame!=self.cfg.algorithm_frame:return
        if self._goal_received and math.dist((gx,gy),self._goal)<getattr(self.cfg,'goal_replay_epsilon',.05):return
        super().set_goal(gx,gy,stamp,frame)
        self._goal_requests=0;self._recovery_goal=None;self._recovery_failed=False
        self._recovery_elapsed=0.;self._last_step=None;self._stall_elapsed=0.
        self._motion_anchor=None;self._reconnect=None
        self._blocked_since=None;self._episode_retries=0;self._episode_anchor=self.centre();self._cooldown=-1e9
        self._pose_fault=False;self._pose_prior=None;self._arbit='ACK';self._session+=1;self._clear_traj_pending=True
        self._velocity_estimator.reset()
    def set_obstacles(self,detections,stamp=None,frame='map',arrival=None,source_age=0.):
        if frame!=self.cfg.algorithm_frame:
            self._t_obs=-1e9;return
        if stamp is not None and self._obs_source is not None and stamp<=self._obs_source:return
        self._obs_source=stamp
        self._t_obs=(time.monotonic() if arrival is None else arrival)-max(0.,source_age)
        objects=[]
        for d in detections:
            x,y,typ,length,width,heading,speed=d[:7];ident=int(d[7]) if len(d)>7 and d[7] is not None else len(objects)
            if not all(math.isfinite(float(v)) for v in (x,y,length,width,heading,speed)) or length<=0 or width<=0:
                self._t_obs=-1e9;return
            if typ.startswith('static') or typ in ('block','unknown'):speed=0.
            shape=d[8] if len(d)>8 else heading
            objects.append(TrackedObstacle(x,y,length,width,shape,speed*math.cos(heading),speed*math.sin(heading),ident,typ=='unknown',0.))
        self._last_received_obs=objects
    def set_wheel_feedback(self,speeds,stamp=None):
        if len(speeds)==4 and all(math.isfinite(float(v)) for v in speeds):
            self._wheel_speeds=list(speeds);self._wheel_ts=time.monotonic() if stamp is None else stamp
    def set_mode_ready(self,mode,arrival=None):
        self._mode_ready=int(mode);self._mode_arrival=time.monotonic() if arrival is None else arrival
    def set_tracker_cmd_speed(self,v,angle=0.,mode=0,arrival=None):
        now=time.monotonic() if arrival is None else arrival
        if self._tracker_cmd_at is not None and mode==0 and self._tracker_mode==0:
            dt=now-self._tracker_cmd_at
            if .005<dt<.15:
                self._tracker_acc=max(-1.2,min(.7,(v-(self._tracker_cmd_v or 0.))/dt))
                self._tracker_steer_rate=max(-math.radians(30),min(math.radians(30),(math.radians(angle)-self._tracker_angle)/dt))
        self._tracker_cmd_at=now
        self._tracker_cmd_v=v;self._tracker_angle=math.radians(angle);self._tracker_mode=mode
    def set_bridge_state(self,state,session=None):
        self._bridge_state=int(state)
        if state in (3,4) and session==self._session and self._arbit=='SPIN_REQUEST':self._last_spin_result='reached' if state==3 else 'timeout'
    def _safe_motion(self,v,w,obs,age):
        delta=math.atan(w*self.geometry.wheelbase/v) if v>.001 else self._tracker_angle
        horizon=max(.8,max(self._v_meas,0.)/1.2+.6)
        rear,_,times=rollout_rear(self.robot.x,self.robot.y,self.robot.theta,max(0.,self._v_meas),self._tracker_angle,[v],[delta],self.geometry,horizon,.05,acc=getattr(self.cfg,'command_acc',.7),steer_rate=math.radians(getattr(self.cfg,'command_steer_rate_deg',30.)),jerk=2.,steer_accel=math.radians(100),initial_acc=self._tracker_acc,initial_steer_rate=self._tracker_steer_rate,delay=.1)
        bound=(max(v,self._v_meas,0.)+abs(w)*self.geometry.radius)*.025
        gap=float(minimum_gaps(to_centres(rear,self.geometry),times,obs,self.geometry,age,bound)[0])
        return gap>=getattr(self.cfg,'safety_margin',.12)
    def _finish(self,info,reason,v=0.,w=0.,road=()):
        if self._clear_traj_pending:self._clear_traj_seq+=1;self._clear_traj_pending=False
        info.update(v_cmd=float(v),w_cmd=float(w),reason=reason,road_base=list(road),traj_mode=int(self._arbit=='SPIN_REQUEST'),
                    arbit=self._arbit,spin_session=self._session,spin_target=self._session_target,
                    traj_clear_seq=self._clear_traj_seq,spin_count=self._episode_retries,goal_spin_requests=self._goal_requests,
                    recovery_goal=self._recovery_goal,recovery_failed=self._recovery_failed,recovery_elapsed=self._recovery_elapsed,
                    ack_bad_cnt=0,v_streak=0,stall_time=self._stall_elapsed)
        info['trap_active']=self._recovery_goal is not None
        self._last_info=info;return info
    def _recovery_heading(self,desired,obs,age,escape):
        """Latch a safe world subgoal; no original-goal distance test.

        Validate rotation and the following straight motion with full rectangles.
        The forward rollout includes mechanical return delay. Every execution
        tick still rechecks current detections; this is not a motion guarantee.
        """
        centre=self.centre();distance=getattr(self.cfg,'recovery_subgoal_distance',1.8)
        offsets=[0.] if not escape else [0.,30.,-30.,60.,-60.,90.,-90.,120.,-120.,150.,-150.,180.]
        candidates=[]
        for offset in offsets:
            yaw=desired+math.radians(offset)
            err=(yaw-self.robot.theta+math.pi)%(2*math.pi)-math.pi
            if abs(err)<math.radians(12):continue
            if not spin_is_safe(*centre,self.robot.theta,yaw,obs,self.geometry,age,
                                speed=getattr(self.cfg,'spin_speed',.6),margin=getattr(self.cfg,'safety_margin',.12)):continue
            if escape:
                progress=np.linspace(0.,distance,41)
                poses=np.column_stack((centre[0]+progress*math.cos(yaw),centre[1]+progress*math.sin(yaw),np.full(41,yaw)))
                times=6.+abs(err)/getattr(self.cfg,'spin_speed',.6)+progress/.4
                if minimum_gaps(poses,times,obs,self.geometry,age,.04)<getattr(self.cfg,'safety_margin',.12):continue
            sub=(centre[0]+distance*math.cos(yaw),centre[1]+distance*math.sin(yaw))
            candidates.append((abs(offset)+.2*abs(math.degrees(err)),yaw,sub))
        return min(candidates,key=lambda q:q[0])[1:] if candidates else None

    def step(self,dt=None,now=None):
        now=time.monotonic() if now is None else now;self._t_now=now
        elapsed=0. if self._last_step is None else max(0.,now-self._last_step);self._last_step=now
        plan_dt=max(.01,min(.2,dt or self.cfg.control_dt))
        obs=self._last_received_obs;self._last_obs=obs
        info={'ok':False,'plan_ok':False,'collision':False,'reached':False,'n_obs':len(obs),'dist_goal':math.inf}
        if not self._goal_received:return self._finish(info,'no_goal')
        if self._pose_fault or now-self._t_loc>self.cfg.localization_timeout:
            self._arbit='ACK';return self._finish(info,'localization_fault')
        if now-self._t_obs>self.cfg.obstacle_timeout:
            self._arbit='ACK';return self._finish(info,'obstacle_timeout')
        centre=self.centre();dg=math.dist(centre,self._goal);info['dist_goal']=dg
        if dg<self.pp.goal_tolerance and abs(self._v_meas)<.04:
            self._arbit='ACK';info['reached']=True;return self._finish(info,'reached')
        age=now-self._t_obs
        self.robot.v=max(0.,self._v_meas);self.robot.omega=self._omega_meas
        self.robot.planning_stamp=now
        if self._arbit=='SPIN_REQUEST':
            safe=spin_is_safe(*centre,self.robot.theta,self._session_target,obs,self.geometry,age,
                              speed=getattr(self.cfg,'spin_speed',.6),margin=getattr(self.cfg,'safety_margin',.12))
            if not safe:self._arbit='ACK';self._cooldown=now+2;return self._finish(info,'spin_safety_abort')
            if self._last_spin_result is not None:
                result=self._last_spin_result;self._last_spin_result=None;self._arbit='ACK';self._cooldown=now+2.;self._clear_traj_pending=True
                return self._finish(info,'spin_'+result)
            if now-self._request_since>getattr(self.cfg,'spin_request_timeout',18.):
                self._arbit='ACK';self._cooldown=now+3.;return self._finish(info,'spin_request_timeout')
            info['ok']=True;return self._finish(info,'spin_active')
        # Mechanical return is a healthy mode transition, not static standstill.
        if self._bridge_state!=0 or self._tracker_mode!=0:return self._finish(info,'mode_return_wait')
        if self._recovery_failed:return self._finish(info,'recovery_exhausted')
        if self._recovery_goal is not None and math.dist(centre,self._recovery_goal)<.3:
            self._recovery_goal=None;self._reconnect=(centre,None);self._stall_elapsed=0.;self._recovery_elapsed=0.
        target=self._recovery_goal or self._goal
        self.planner.current_steer=self._tracker_angle;self.planner.obstacle_age=age;self.planner.params.dt=plan_dt
        self.planner.current_acc=self._tracker_acc;self.planner.current_steer_rate=self._tracker_steer_rate
        ok,v,w=self.planner.plan(self.robot,target,obs,None)
        telemetry=getattr(self.planner,'telemetry',{});info.update(plan_ok=ok,planner=telemetry)
        reason=telemetry.get('reason','blocked')
        # Do not advance clocks while input/computation is bad or while yielding.
        valid=reason in ('motion','goal_braking','dynamic_braking','dynamic_wait','blocked','orientation_stall')
        if not valid:return self._finish(info,reason)
        if reason in ('goal_braking','dynamic_braking','dynamic_wait'):
            if not self._safe_motion(v,w,obs,age):return self._finish(info,'execution_collision_guard')
            info['ok']=True
            return self._finish(info,reason,v,w,getattr(self.planner,'reference',[]))
        if self._motion_anchor is None or math.dist(centre,self._motion_anchor)>.12:
            self._motion_anchor=centre;self._motion_at=now
        self._standstill_elapsed=0. if self._motion_at==now else self._standstill_elapsed+elapsed
        actual_stall=self._standstill_elapsed>getattr(self.cfg,'motion_stall_time',3.)
        stagnant=(not ok or v<.005 or actual_stall)
        self._stall_elapsed=self._stall_elapsed+elapsed if stagnant else 0.
        self._stall_time=self._stall_elapsed
        if self._recovery_goal is not None:
            self._recovery_elapsed+=elapsed
            if self._recovery_elapsed>getattr(self.cfg,'trap_max_duration',45.):
                self._recovery_failed=True;return self._finish(info,'recovery_exhausted')
        local=telemetry.get('guide_target',target)
        desired=math.atan2(local[1]-centre[1],local[0]-centre[0])
        err=(desired-self.robot.theta+math.pi)%(2*math.pi)-math.pi
        near_turn=dg<getattr(self.cfg,'near_goal_spin_distance',3.5) and abs(err)>math.radians(getattr(self.cfg,'near_goal_spin_angle_deg',75.))
        blocked=self._stall_elapsed>=getattr(self.cfg,'blocked_spin_time',1.5)
        if (near_turn or blocked) and now>=self._cooldown and getattr(self.cfg,'enable_trap',True):
            if self._episode_retries>=getattr(self.cfg,'recovery_max_retries',2) or self._goal_requests>=getattr(self.cfg,'goal_max_spin_requests',4):
                if stagnant:self._recovery_failed=True;return self._finish(info,'recovery_exhausted')
            else:
                escape=blocked and (reason=='blocked' or actual_stall or self._recovery_goal is not None)
                choice=self._recovery_heading(desired,obs,age,escape)
                if choice is not None:
                    yaw,sub=choice
                    if escape:self._recovery_goal=sub;self._recovery_elapsed=0.;self._reconnect=None
                    self._arbit='SPIN_REQUEST';self._session+=1;self._session_target=yaw;self._request_since=now
                    self._episode_retries+=1;self._goal_requests+=1;self._stall_elapsed=0.;self._clear_traj_pending=True;self._last_spin_result=None
                    info['ok']=True;return self._finish(info,'spin_requested')
                elif blocked and reason=='blocked':
                    self._recovery_failed=True;return self._finish(info,'recovery_no_safe_heading')
        if not ok or v<.005:return self._finish(info,reason)
        if not self._safe_motion(v,w,obs,age):return self._finish(info,'execution_collision_guard')
        if self._reconnect is not None and telemetry.get('forecast_progress',0.)>.15:
            anchor,waypoint=self._reconnect
            if waypoint is None:waypoint=tuple(local);self._reconnect=(anchor,waypoint)
            dx,dy=waypoint[0]-anchor[0],waypoint[1]-anchor[1];length=max(math.hypot(dx,dy),1e-6)
            projected=((centre[0]-anchor[0])*dx+(centre[1]-anchor[1])*dy)/length
            if projected>.4:self._episode_retries=0;self._reconnect=None
        info['ok']=True;self.state.v,self.state.omega=v,w
        self.planner.set_last_cmd(v,w)
        return self._finish(info,'motion',v,w,getattr(self.planner,'reference',[]))
