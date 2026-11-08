"""Bounded Ackermann RDA-DWA execution for the real vehicle.

Samples executable steering targets with a persistent static visibility guide.
Known occupancy uses the full forecast; unknown occupancy checks the committed
action and complete braking backup. Execution rechecks fresh observations.
No ROS.
"""
import math,time,heapq
from dataclasses import replace
import numpy as np
from .enhanced_dwa import EnhancedDWA
from .motion_safety import VehicleGeometry,rollout_rear,to_centres,minimum_gaps,obstacle_from_object,rollout_stop,stopping_time_limit

class RealtimeAckermannPlanner(EnhancedDWA):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.geometry=VehicleGeometry(self.cc.ack.length,self.cc.ack.width,self.cc.ack.wheelbase)
        self.reference=[];self.telemetry={};self._guide=[];self._guide_goal=None
        self.current_steer=0.;self.obstacle_age=0.
        self.current_acc=0.;self.current_steer_rate=0.;self._encounter=None;self._provisional=None
    def _guide_target(self,start,goal,obstacles,deadline):
        # Guide boxes are a separate geometric heuristic; the final OBB rollout
        # ALWAYS validates physical length, width, orientation and safety margin.
        # Build the heuristic in a goal-aligned frame fixed for this goal.
        # Map yaw must not change the required manoeuvre for an identical scene.
        if self._guide_goal!=tuple(goal):
            self._guide=[];self._guide_goal=tuple(goal);self._guide_origin=start
            self._guide_axis=math.atan2(goal[1]-start[1],goal[0]-start[0])
        ox,oy=self._guide_origin;basis_c,basis_s=math.cos(self._guide_axis),math.sin(self._guide_axis)
        def inward(p):
            dx,dy=p[0]-ox,p[1]-oy;return (round(basis_c*dx+basis_s*dy,12),round(-basis_s*dx+basis_c*dy,12))
        def outward(p):return (ox+basis_c*p[0]-basis_s*p[1],oy+basis_s*p[0]+basis_c*p[1])
        start,goal=inward(start),inward(goal)
        obstacles=[replace(o,x=inward((o.x,o.y))[0],y=inward((o.x,o.y))[1],heading=o.heading-self._guide_axis,
                           vx=basis_c*o.vx+basis_s*o.vy,vy=-basis_s*o.vx+basis_c*o.vy) for o in obstacles]
        boxes=[];pad=self.geometry.width/2+getattr(self.params,'safety_margin',.12)+getattr(self.params,'guide_extra',.08)
        for o in obstacles:
            if o.speed>.05 or o.uncertain:continue
            c,s=abs(math.cos(o.heading)),abs(math.sin(o.heading))
            hx=c*o.length/2+s*o.width/2+pad;hy=s*o.length/2+c*o.width/2+pad
            if math.hypot(o.x-start[0],o.y-start[1])<8:boxes.append((o.x-hx,o.y-hy,o.x+hx,o.y+hy))
        if len(boxes)>8:boxes=sorted(boxes,key=lambda b:math.hypot((b[0]+b[2])/2-start[0],(b[1]+b[3])/2-start[1]))[:8]
        def visible(a,b):
            for x0,y0,x1,y1 in boxes:
                lo,hi=0.,1.
                for p,d,mn,mx in [(a[0],b[0]-a[0],x0,x1),(a[1],b[1]-a[1],y0,y1)]:
                    if abs(d)<1e-9:
                        if p<=mn+1e-5 or p>=mx-1e-5:lo=2.;break
                    else:
                        aa,bb=sorted(((mn+1e-5-p)/d,(mx-1e-5-p)/d));lo=max(lo,aa);hi=min(hi,bb)
                if lo<hi and hi>0 and lo<1:return False
            return True
        while self._guide and (math.dist(start,self._guide[0])<.65 or (len(self._guide)>1 and visible(start,self._guide[1]))):self._guide.pop(0)
        if self._guide:return outward(self._guide[0])
        if visible(start,goal):self._guide=[];return outward(goal)
        nodes=[start,goal]
        for x0,y0,x1,y1 in boxes:nodes.extend([(x0-.05,y0-.05),(x0-.05,y1+.05),(x1+.05,y0-.05),(x1+.05,y1+.05)])
        dist=[math.inf]*len(nodes);dist[0]=0.;prev={};queue=[(0.,0)]
        while queue and time.perf_counter()<deadline:
            value,i=heapq.heappop(queue)
            if value!=dist[i]:continue
            if i==1:break
            for j,b in enumerate(nodes):
                if i==j or not visible(nodes[i],b):continue
                vv=round(value+math.dist(nodes[i],b),10)
                if vv<dist[j]:dist[j]=vv;prev[j]=i;heapq.heappush(queue,(vv,j))
        if 1 in prev:
            route=[];i=1
            while i:route.append(nodes[i]);i=prev[i]
            self._guide=list(reversed(route));return outward(self._guide[0])
        return outward(goal)
    def _closing_actor(self,centre,yaw,speed,obs,stamp=None):
        """Latch the side of a closing encounter until the actor passes."""
        stamp=time.monotonic() if stamp is None else stamp
        forward=np.array([math.cos(yaw),math.sin(yaw)])
        if self._encounter is not None:
            e=self._encounter
            actor=next((o for o in obs if o.ident==e['id'] and not o.uncertain),None)
            if actor is not None:
                rel=np.array([actor.x-centre[0],actor.y-centre[1]])
                if rel@e['axis']>-(self.geometry.length+actor.length):return actor
            uncertain=next((o for o in obs if o.ident==e['id'] and o.uncertain),None)
            if uncertain is not None:
                self._provisional=dict(id=e['id'],axis=e['axis'],origin=e['origin'],side=e['side'],
                                       seen_at=stamp,obstacle_centre=(uncertain.x,uncertain.y))
            self._encounter=None
        choices=[]
        for o in obs:
            if o.uncertain or o.speed<.2:continue
            r=np.array([o.x-centre[0],o.y-centre[1]])
            velocity=np.array([o.vx,o.vy])-speed*forward
            along=r@forward;lateral=abs(r[0]*forward[1]-r[1]*forward[0])
            closing=-velocity@forward
            if along>.3 and closing>.2 and np.array([o.vx,o.vy])@forward<-.15 and lateral<1.4+o.width/2:
                ttc=along/closing
                if ttc<7.:choices.append((ttc,o))
        if not choices:return None
        actor=min(choices,key=lambda q:q[0])[1]
        self._encounter=dict(id=actor.ident,axis=forward,origin=np.asarray(centre),side=self._provisional['side'] if self._provisional and self._provisional['id']==actor.ident else 0)
        return actor

    def _unknown_in_corridor(self,centre,yaw,obs,stamp=None):
        stamp=time.monotonic() if stamp is None else stamp
        forward=np.array([math.cos(yaw),math.sin(yaw)])
        relevant=[]
        for o in obs:
            if not o.uncertain:continue
            rel=np.array([o.x-centre[0],o.y-centre[1]])
            along=float(rel@forward);lateral=abs(rel[0]*forward[1]-rel[1]*forward[0])
            if .3<along<6. and lateral<self.geometry.width/2+max(o.length,o.width)/2+.3:
                relevant.append((along,o))
        if not relevant:self._provisional=None;return None
        actor=min(relevant,key=lambda q:q[0])[1]
        prior=self._provisional
        same_patch=prior is not None and 0.<=stamp-prior['seen_at']<.8 and math.dist(prior['obstacle_centre'],(actor.x,actor.y))<.8
        if prior is None or (prior['id']!=actor.ident and not same_patch):
            self._provisional=dict(id=actor.ident,axis=forward,origin=np.asarray(centre),side=0)
        self._provisional.update(id=actor.ident,seen_at=stamp,obstacle_centre=(actor.x,actor.y))
        return self._provisional

    def _future_refuge(self,poses,times,obs,margin):
        """Terminal parked refuge against each known CV actor's future ray.

        A bounded prediction is not a guarantee for actors that change course.
        Unknown actors continue to use the full reachable envelope instead.
        """
        safe=np.ones(len(poses),dtype=bool)
        end=poses[:,-1,:2]
        for o in obs:
            if o.uncertain or o.speed<.1:continue
            velocity=np.array([o.vx,o.vy]);start=np.array([o.x,o.y])+velocity*(times[-1]+self.obstacle_age)
            rel=end-start;u=np.maximum(0.,rel@velocity/(o.speed**2))
            distance=np.linalg.norm(rel-u[:,None]*velocity,axis=1)
            safe &= distance>=self.geometry.radius+math.hypot(o.length,o.width)/2+margin
        return safe

    def plan(self,robot,goal,obstacles,trap=None):
        begin=time.perf_counter();deadline=begin+getattr(self.params,'planning_budget_s',.25)
        self.reference=[]
        obs=[o if hasattr(o,'vx') else obstacle_from_object(o) for o in obstacles]
        if len(obs)>96:self.telemetry={'reason':'obstacle_capacity'};return False,0.,0.
        centre=(robot.x+self.geometry.rear_offset*math.cos(robot.theta),robot.y+self.geometry.rear_offset*math.sin(robot.theta))
        local=self._guide_target(centre,goal,obs,min(deadline,begin+.015))
        dt=max(.02,min(.2,self.params.dt));acc=getattr(self.params,'command_acc',.7)
        dec=getattr(self.params,'command_dec',1.2);jerk=getattr(self.params,'command_jerk',2.)
        steer_rate=getattr(self.params,'command_steer_rate',math.radians(30))
        steer_acc=getattr(self.params,'command_steer_accel',math.radians(100))
        delay=getattr(self.params,'actuator_delay_s',.1)
        distance=max(0.,math.dist(centre,goal)-.18)
        # Include positive acceleration unwind and command delay before braking.
        brake_delay=delay+.25+acc/jerk
        goal_cap=max(0.,math.sqrt((dec*brake_delay)**2+2*dec*distance)-dec*brake_delay)
        forecast=getattr(self.params,'velocity_horizon_targets',False)
        vmax=min(self.params.max_linear_velocity,getattr(self.params,'cruise_velocity',1.),goal_cap)
        if not forecast:vmax=min(vmax,max(0.,robot.v)+acc*dt)
        actor=self._closing_actor(centre,robot.theta,robot.v,obs,getattr(robot,'planning_stamp',None)) if getattr(self.params,'staged_avoidance',True) else None
        provisional=self._unknown_in_corridor(centre,robot.theta,obs,getattr(robot,'planning_stamp',None)) if actor is None else None
        if actor is not None:
            e=self._encounter;normal=np.array([-e['axis'][1],e['axis'][0]])
            # A fixed world direction prevents left/right chattering. The final
            # choice still has to pass time-indexed physical collision checks.
            side=e['side'] or 1
            local=tuple(e['origin']+1.1*e['axis']+side*1.5*normal)
            vmax=min(self.params.max_linear_velocity,getattr(self.params,'cruise_velocity',1.))
        if provisional is not None:
            normal=np.array([-provisional['axis'][1],provisional['axis'][0]])
            local=tuple(provisional['origin']+1.1*provisional['axis']+(provisional['side'] or 1)*1.5*normal)
        horizon=max(self.params.sim_time,5. if actor is not None else 0.)
        times=np.linspace(0.,horizon,int(math.ceil(horizon/.05))+1)
        # Forecast targets are desired servo inputs, not instantaneous actual
        # velocities. Include intermediate lower targets for gradual braking;
        # sampling only zero and near-current speed causes stop/start pulses.
        velocities=np.unique(np.r_[0.,np.linspace(0. if forecast else min(vmax,max(0.,robot.v-dec*dt)),vmax,9),min(vmax,max(0.,robot.v))])
        deltas=np.unique(np.r_[np.linspace(-self.cc.ack.delta_max,self.cc.ack.delta_max,17),self.current_steer,0.])
        vv,dd=np.meshgrid(velocities,deltas,indexing='ij');vs=vv.ravel();ds=dd.ravel()
        vs=np.minimum(vs,np.sqrt(getattr(self.params,'lateral_acc_max',.6)*self.geometry.wheelbase/np.maximum(np.abs(np.tan(ds)),.001)))
        vtargets=np.repeat(vs[:,None],len(times),axis=1)
        dtargets=np.repeat(ds[:,None],len(times),axis=1)
        staged=np.zeros(len(vs),dtype=bool);sides=np.zeros(len(vs),dtype=int);continuation=np.zeros(len(vs),dtype=bool)
        if actor is not None:
            sv=[];sd=[];ss=[]
            for side in (-1,1):
                if self._encounter['side'] and side!=self._encounter['side']:continue
                for cruise in (.45,.7,min(.95,vmax)):
                    for turn_time in (.7,1.1,1.5):
                        for brake_time in (2.4,3.2):
                            vprofile=np.where(times<brake_time,cruise,0.)
                            # Turn out, countersteer, then settle before parking.
                            dprofile=np.where(times<turn_time,side*self.cc.ack.delta_max,
                                      np.where(times<2.*turn_time,-side*self.cc.ack.delta_max,0.))
                            sv.append(vprofile);sd.append(dprofile);ss.append(side)
            e=self._encounter
            elapsed=max(0.,getattr(robot,'planning_stamp',time.monotonic())-e.get('schedule_stamp',getattr(robot,'planning_stamp',time.monotonic())))
            existing='schedule_v' in e and elapsed<e['schedule_times'][-1]
            if existing:
                sv.append(np.interp(times+elapsed,e['schedule_times'],e['schedule_v'],right=0.))
                sd.append(np.interp(times+elapsed,e['schedule_times'],e['schedule_d'],right=0.))
                ss.append(e['side'])
            continuation=np.r_[continuation,np.zeros(len(sv),dtype=bool)]
            if existing:continuation[-1]=True
            vtargets=np.vstack((vtargets,sv));dtargets=np.vstack((dtargets,sd))
            staged=np.r_[staged,np.ones(len(sv),dtype=bool)];sides=np.r_[sides,ss]
        rear,speeds,times,steering,accelerations,rates=rollout_rear(
            robot.x,robot.y,robot.theta,robot.v,self.current_steer,vtargets,dtargets,self.geometry,
            horizon,.05,acc,dec,steer_rate,jerk,steer_acc,self.current_acc,self.current_steer_rate,delay,True)
        centres=to_centres(rear,self.geometry)
        maximum=max(vmax,robot.v,.01)
        rates_of_motion=speeds+np.abs(speeds*np.tan(steering)/self.geometry.wheelbase)*self.geometry.radius
        # Each sample bounds its adjacent intervals using that candidate's
        # actual ramp. A global cruise bound can reject a safe stationary
        # start near a wall before the car has accelerated at all.
        sweep=np.maximum(rates_of_motion,np.c_[rates_of_motion[:,1:],rates_of_motion[:,-1]])*.025+.002
        margin=getattr(self.params,'safety_margin',.12)
        known=[o for o in obs if not o.uncertain];unknown=[o for o in obs if o.uncertain]
        gaps=minimum_gaps(centres,times,known,self.geometry,self.obstacle_age,sweep)
        if unknown:
            prefix=minimum_gaps(centres,times,unknown,self.geometry,self.obstacle_age,sweep,valid_until=.4)
            gaps=np.minimum(gaps,prefix)
        dynamic=any(o.speed>.1 or o.uncertain for o in obs)
        stop_limits=None;unknown_stop=None
        if dynamic and (forecast or unknown):
            stop,st,stop_speeds=rollout_stop(robot.x,robot.y,robot.theta,robot.v,self.current_steer,
                vtargets[:,0],dtargets[:,0],self.geometry,horizon,.05,reaction=.4,jerk=jerk,
                steer_accel=steer_acc,initial_acc=self.current_acc,initial_steer_rate=self.current_steer_rate,return_speeds=True)
            stop_rate=stop_speeds*(1.+math.tan(self.cc.ack.delta_max+steer_rate**2/(2*steer_acc))/self.geometry.wheelbase*self.geometry.radius)
            stop_sweep=np.maximum(stop_rate,np.c_[stop_rate[:,1:],stop_rate[:,-1]])*.025+.002
            stop_gaps=minimum_gaps(to_centres(stop,self.geometry),st,known,self.geometry,self.obstacle_age,stop_sweep)
            if unknown:
                stop_limits=stopping_time_limit(stop_speeds,st)
                unknown_stop=minimum_gaps(to_centres(stop,self.geometry),st,unknown,self.geometry,self.obstacle_age,stop_sweep,valid_until=stop_limits)
                # Unknown protection is required for every selected candidate,
                # including a staged refuge with a known actor.
                gaps=np.minimum(gaps,unknown_stop)
            # For a head-on escape, stopping in the actor's path is insufficient.
            # Only a complete staged path ending in a checked lateral refuge
            # may use continuation instead of the ordinary stop-now backup.
            refuge=staged&(speeds[:,-1]<.04)&self._future_refuge(centres,times,obs,margin)
            gaps=np.where(refuge,gaps,np.minimum(gaps,stop_gaps))
            if actor is not None:gaps=np.where(staged&~refuge,-np.inf,gaps)
        feasible=gaps>=margin
        movement=np.max(speeds,axis=1)>.025
        static_motion=bool(np.any(feasible&movement))
        if dynamic:
            static_obs=[o for o in obs if o.speed<=.1 and not o.uncertain]
            static_gaps=minimum_gaps(centres,times,static_obs,self.geometry,self.obstacle_age,sweep)
            static_motion=bool(np.any((static_gaps>=margin)&movement))
        def finite_gap(value):return float(value) if math.isfinite(value) else None
        feasibility=dict(candidates=len(vtargets),feasible=int(sum(feasible)),motion_feasible=bool(np.any(feasible&movement)),
             obstacle_count=len(obs),unknown_count=sum(o.uncertain for o in obs),moving_count=sum(o.speed>.1 for o in obs),
             best_gap=finite_gap(np.max(gaps)),encounter_id=actor.ident if actor else None,
             avoidance_side=self._encounter['side'] if actor else 0,staged_feasible=int(sum(feasible&staged)),unknown_policy='action_and_brake',unknown_reaction_s=.4)
        if time.perf_counter()>deadline:
            self.telemetry=dict(feasibility,reason='planning_budget',duration_s=time.perf_counter()-begin);return False,0.,0.
        end=centres[:,-1,:];d0=math.dist(centre,local);dist=np.hypot(end[:,0]-local[0],end[:,1]-local[1])
        desired=math.atan2(local[1]-centre[1],local[0]-centre[0]);err=(desired-end[:,2]+math.pi)%(2*math.pi)-math.pi
        score=3*(d0-dist)-getattr(self.params,'heading_weight',.3)*np.abs(err)+.15*np.max(speeds,axis=1)
        score+=.15*np.minimum(gaps,.4)
        score-=getattr(self.params,'steer_weight',.4)*np.mean((dtargets/self.cc.ack.delta_max)**2,axis=1)
        score-=getattr(self.params,'steer_change_weight',.12)*((dtargets[:,0]-self.current_steer)/self.cc.ack.delta_max)**2
        if provisional is not None:
            axis=provisional['axis'];normal=np.array([-axis[1],axis[0]])
            for side in (-1,1):
                target=provisional['origin']+1.1*axis+side*1.5*normal
                mask=(dtargets[:,0]>=0.) if side==1 else (dtargets[:,0]<0.)
                if provisional['side'] and side!=provisional['side']:score[mask]-=100.
                else:score[mask]+=3*(dist[mask]-np.linalg.norm(end[mask,:2]-target,axis=1))
            score+=np.where(dtargets[:,0]>=0.,.01,0.)
        if actor is not None and not self._encounter['side']:
            # Equal-cost ties have a stable preference; blocked sides remain
            # available until a physically feasible choice is established.
            score+=np.where(sides==1,.03,0.)
            # Score each candidate against the refuge on its own side.
            axis=self._encounter['axis'];normal=np.array([-axis[1],axis[0]])
            for side in (-1,1):
                target=self._encounter['origin']+1.1*axis+side*1.5*normal
                mask=sides==side
                candidate_dist=np.linalg.norm(end[:,:2]-target,axis=1)
                score[mask]+=3*(dist[mask]-candidate_dist[mask])
        # Prefer a still-feasible continuation over repeatedly restarting turn-out.
        if np.any(continuation&feasible):score[continuation&feasible]+=100.
        score[~feasible]=-np.inf
        if not np.any(feasible):
            reason='dynamic_yield' if static_motion and dynamic else 'blocked'
            self.telemetry=dict(feasibility,reason=reason,duration_s=time.perf_counter()-begin,guide_target=local,static_motion_feasible=static_motion,
                 yield_cause='no_validated_lateral_refuge' if actor is not None else ('unknown_reachable_envelope' if any(o.uncertain for o in obs) else None))
            return False,0.,0.
        k=int(np.flatnonzero(score>=np.max(score)-1e-10)[0])
        cruise=float(vtargets[k,0]);delta=float(dtargets[k,0]);v=cruise
        if provisional is not None and cruise>.005 and abs(delta)>.05:
            provisional['side']=1 if delta>0. else -1
            feasibility['provisional_avoidance_id']=provisional['id'];feasibility['avoidance_side']=provisional['side']
        if staged[k] and actor is not None:
            self._encounter['side']=int(sides[k]);feasibility['avoidance_side']=int(sides[k])
            if not continuation[k]:
                self._encounter.update(schedule_v=vtargets[k].copy(),schedule_d=dtargets[k].copy(),
                    schedule_times=times.copy(),schedule_stamp=getattr(robot,'planning_stamp',time.monotonic()))
        moving=bool(np.max(speeds[k])>.025)
        if moving or cruise>.005:
            self.reference=[(float(p[0]),float(p[1]),float(p[2]),float(vt),float(t),float(d))
                            for p,vt,t,d in zip(rear[k],vtargets[k],times,dtargets[k])]
        reason='motion' if cruise>.005 else (('dynamic_braking' if actor is not None else 'goal_braking') if robot.v>.025 and moving else
               ('dynamic_wait' if actor is not None else ('orientation_stall' if feasibility['motion_feasible'] else ('dynamic_yield' if static_motion and dynamic else 'blocked'))))
        self.telemetry=dict(feasibility,reason=reason,duration_s=time.perf_counter()-begin,
             min_gap=finite_gap(gaps[k]),guide_target=local,selected_delta=delta,staged_selected=bool(staged[k]),
             unknown_stop_horizon_s=float(stop_limits[k]) if stop_limits is not None else None,
             unknown_stop_min_gap=finite_gap(unknown_stop[k]) if unknown_stop is not None else None,
             static_motion_feasible=static_motion,forecast_progress=float(d0-dist[k]),cruise=cruise)
        return True,v,v*math.tan(delta)/self.geometry.wheelbase
