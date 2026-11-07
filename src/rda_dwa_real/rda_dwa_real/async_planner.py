"""Single spawned planning process; main ROS callbacks never wait for a plan."""
import copy,time,multiprocessing,queue,math

def _work(input_queue,output_queue,planner):
    output_queue.put(('ready',None))
    while True:
        item=input_queue.get()
        if item is None:return
        seq,stamp,robot,goal,obs,trap,steer,age,plan_dt,*actuator=item
        try:
            started=time.monotonic();planner.current_steer=steer;planner.obstacle_age=age
            planner.current_acc,planner.current_steer_rate=actuator if actuator else (0.,0.)
            planner.params.dt=plan_dt
            result=planner.plan(robot,goal,obs,trap)
            packet=(seq,stamp,robot,goal,result,planner.reference,dict(planner.telemetry,worker_duration_s=time.monotonic()-started,worker_queue_delay_s=max(0.,started-stamp),sampled_control_dt_s=plan_dt,sampled_pose=[robot.x,robot.y,robot.theta],sampled_velocity=robot.v,sampled_steer=steer,sampled_obstacle_age_s=age))
        except Exception as exc:packet=(seq,stamp,robot,goal,(False,0.,0.),[],{'reason':'planning_exception','error':type(exc).__name__})
        output_queue.put(('result',packet))

class AsyncPlanner:
    def __init__(self,planner,max_age=.8):
        if not math.isfinite(max_age) or max_age<=0:raise ValueError('planning result max age must be positive')
        self.base=planner;self.max_age=max_age;self.reference=[];self.telemetry={}
        self.current_steer=0.;self.obstacle_age=0.;self._pending=False;self._ready=False;self._seq=0;self._result=None
        ctx=multiprocessing.get_context('spawn');self._input=ctx.Queue(1);self._output=ctx.Queue(2)
        self._process=ctx.Process(target=_work,args=(self._input,self._output,planner),daemon=True);self._process.start()
    def __getattr__(self,name):return getattr(self.base,name)
    def plan(self,robot,goal,obs,trap=None):
        now=time.monotonic()
        while True:
            try:kind,payload=self._output.get_nowait()
            except queue.Empty:break
            if kind=='ready':self._ready=True
            else:self._pending=False;self._result=payload
        if self._ready and not self._pending and self._process.is_alive():
            self._seq+=1;item=(self._seq,now,copy.copy(robot),tuple(goal),copy.deepcopy(obs),copy.deepcopy(trap),self.current_steer,self.obstacle_age,self.params.dt,getattr(self,'current_acc',0.),getattr(self,'current_steer_rate',0.))
            try:self._input.put_nowait(item);self._pending=True
            except queue.Full:pass
        self.reference=[]
        if not self._process.is_alive():self.telemetry={'reason':'planning_worker_failed'};return False,0.,0.
        if self._result is None:self.telemetry={'reason':'planning_pending','worker_ready':self._ready,'request_sequence':self._seq,'pending':self._pending};return False,0.,0.
        seq,stamp,pose,old_goal,result,reference,telemetry=self._result
        err=(robot.theta-pose.theta+math.pi)%(2*math.pi)-math.pi
        age=now-stamp
        displacement=math.hypot(robot.x-pose.x,robot.y-pose.y)
        # Legitimate progress on the checked prefix is not a localization jump.
        # Restrict projections by elapsed time so a loop or a distant tail
        # cannot make a displaced pose appear valid.
        deviation=displacement;heading_error=err;progress_time=0.
        for a,b in zip(reference,reference[1:]):
            if len(a)<5 or len(b)<5 or a[4]>age+.15:break
            dx,dy=b[0]-a[0],b[1]-a[1];length2=dx*dx+dy*dy
            u=max(0.,min(1.,((robot.x-a[0])*dx+(robot.y-a[1])*dy)/max(length2,1e-12)))
            if b[4]>a[4]:u=min(u,max(0.,(age+.15-a[4])/(b[4]-a[4])))
            distance=math.hypot(robot.x-a[0]-u*dx,robot.y-a[1]-u*dy)
            if distance<deviation:
                deviation=distance;progress_time=a[4]+u*(b[4]-a[4])
                yaw=a[2]+u*((b[2]-a[2]+math.pi)%(2*math.pi)-math.pi)
                heading_error=(robot.theta-yaw+math.pi)%(2*math.pi)-math.pi
        stale_causes=[]
        if old_goal!=tuple(goal):stale_causes.append('goal_changed')
        if age>self.max_age+1e-6:stale_causes.append('result_age')
        if deviation>.18:stale_causes.append('pose_displacement')
        if abs(heading_error)>math.radians(12):stale_causes.append('yaw_change')
        details=dict(telemetry,worker_reason=telemetry.get('reason'),result_age_s=age,max_result_age_s=self.max_age,sequence=seq,goal_changed=old_goal!=tuple(goal),pose_displacement_m=displacement,yaw_change_rad=err,path_deviation_m=deviation,path_heading_error_rad=heading_error,path_progress_time_s=progress_time,pending=self._pending,stale_causes=stale_causes)
        if stale_causes:
            self.telemetry=dict(details,reason='planning_result_stale');return False,0.,0.
        self.reference=reference;self.telemetry=details
        return result
    def close(self):
        if self._process.is_alive():
            try:self._input.put_nowait(None)
            except queue.Full:pass
            self._process.join(timeout=.2)
            if self._process.is_alive():self._process.terminate();self._process.join(timeout=.2)
        self._input.close();self._output.close()
