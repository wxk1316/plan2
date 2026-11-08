"""One bounded perception process; ROS callbacks keep receiving current poses.

The single waiting input slot is replaced by newer clouds. Results retain
original acquisition times; failed/expired observations never publish an empty
free scene. No ROS objects or publishers cross the process boundary.
"""
import multiprocessing
import math
import os
import queue
import time
from types import SimpleNamespace
from .perception_core import LidarPerceptionCore,point_cloud_xyz


def cloud_packet(msg):
    return dict(sec=int(msg.header.stamp.sec),nanosec=int(msg.header.stamp.nanosec),
                width=int(msg.width),height=int(msg.height),point_step=int(msg.point_step),
                row_step=int(getattr(msg,'row_step',msg.width*msg.point_step)),
                is_bigendian=bool(getattr(msg,'is_bigendian',False)),data=bytes(msg.data),
                fields=[(f.name,int(f.offset),int(f.datatype),int(getattr(f,'count',1))) for f in msg.fields])


def unpack_cloud(packet):
    return SimpleNamespace(**{k:v for k,v in packet.items() if k!='fields'},
        fields=[SimpleNamespace(name=n,offset=o,datatype=t,count=c) for n,o,t,c in packet['fields']])


def replace_pending(channel,item):
    """Nonblocking, bounded latest item; feeder races may skip one arrival."""
    try:channel.put_nowait(item);return True
    except queue.Full:pass
    try:channel.get_nowait()
    except queue.Empty:return False
    try:channel.put_nowait(item);return True
    except queue.Full:return False


class PerceptionWorkerCore:
    def __init__(self,perception_params,tracker,reset_gap=2.,max_source_age=.4):
        self.perception=LidarPerceptionCore(perception_params);self.tracker=tracker
        if not all(math.isfinite(v) and v>0 for v in (reset_gap,max_source_age)):raise ValueError('invalid perception time limits')
        self.reset_gap=reset_gap;self.max_source_age=max_source_age;self.last_stamp=None;self.last_tracking_stamp=None

    def process(self,job,started=None):
        sequence,received,cloud,pose,metrics=job
        begin=time.monotonic() if started is None else started;metrics=dict(metrics)
        stamp=cloud['sec']+cloud['nanosec']*1e-9;tracks=None
        # ROS age at capture plus elapsed monotonic time remains valid across
        # a later ROS clock jump. Never renew it with a publication timestamp.
        age=metrics['source_age_at_receive_s']+max(0.,begin-received)
        reason='processing'
        try:
            if self.last_stamp is not None and stamp<=self.last_stamp:reason='cloud_source_stale_or_repeated'
            elif age>self.max_source_age:reason='cloud_expired_in_queue'
            else:
                dt=stamp-self.last_stamp if self.last_stamp is not None else self.tracker.dt
                self.last_stamp=stamp
                metrics.update(source_dt_s=dt,track_reset=False,track_reset_gap_s=self.reset_gap)
                phase=time.monotonic();xyz=point_cloud_xyz(unpack_cloud(cloud))
                metrics['parse_duration_s']=time.monotonic()-phase
                phase=time.monotonic();detections,m=self.perception.process(xyz,stamp,pose)
                metrics.update(m,filter_cluster_duration_s=time.monotonic()-phase)
                reason=m['reason']
                if m['valid']:
                    tracking_dt=stamp-self.last_tracking_stamp if self.last_tracking_stamp is not None else self.tracker.dt
                    reset=self.last_tracking_stamp is not None and tracking_dt>self.reset_gap
                    if reset:self.tracker.reset()
                    self.last_tracking_stamp=stamp
                    metrics.update(track_reset=reset,tracking_dt_s=tracking_dt)
                    import math
                    c,s=math.cos(pose[2]),math.sin(pose[2])
                    world=[(pose[0]+c*d[0]-s*d[1],pose[1]+s*d[0]+c*d[1],d[2],d[3],d[4],d[5]+pose[2]) for d in detections]
                    metrics['detections_world']=world
                    phase=time.monotonic();tracks=self.tracker.step(world,dt=max(.005,tracking_dt))
                    metrics.update(kf_duration_s=time.monotonic()-phase,track_outputs=tracks,track_count=len(tracks),unknown_count=sum(t[5]=='unknown' for t in tracks))
                    reason='published' if detections else 'empty_cloud_predicted_tracks'
        except (ValueError,TypeError,BufferError) as exc:
            reason='malformed_cloud';metrics['exception']=type(exc).__name__
        except Exception as exc:
            reason='perception_exception';metrics['exception']=type(exc).__name__
        metrics.update(reason=reason,worker_duration_s=time.monotonic()-begin,
                       worker_queue_delay_s=max(0.,begin-received))
        metrics['duration_s']=metrics['worker_duration_s']
        return sequence,received,cloud['sec'],cloud['nanosec'],tracks,metrics


def _work(incoming,outgoing,params,tracker,reset_gap,max_age):
    core=PerceptionWorkerCore(params,tracker,reset_gap,max_age);outgoing.put(('ready',None))
    while True:
        job=incoming.get()
        if job is None:return
        # Pick the newest waiting observation before beginning the next frame.
        while True:
            try:newer=incoming.get_nowait()
            except queue.Empty:break
            if newer is None:return
            job=newer
        result=core.process(job)
        replace_pending(outgoing,('result',result))


class AsyncPerception:
    def __init__(self,params,tracker,reset_gap=2.,max_source_age=.4):
        ctx=multiprocessing.get_context('spawn');self.incoming=ctx.Queue(1);self.outgoing=ctx.Queue(2)
        self.process=ctx.Process(target=_work,args=(self.incoming,self.outgoing,params,tracker,reset_gap,max_source_age),daemon=True)
        self.ready=False;self.submitted=0;self.skipped=0
        # Bound numerical-library threads in this child only. Many small
        # matrices should not compete with localization/control on Jetson.
        keys=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')
        prior={key:os.environ.get(key) for key in keys}
        try:
            for key in keys:os.environ[key]='1'
            self.process.start()
        finally:
            for key,value in prior.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value

    def submit(self,job):
        if not self.process.is_alive():return False
        accepted=replace_pending(self.incoming,job)
        self.submitted+=int(accepted);self.skipped+=int(not accepted)
        return accepted

    def poll(self):
        latest=None
        while True:
            try:kind,payload=self.outgoing.get_nowait()
            except queue.Empty:break
            if kind=='ready':self.ready=True
            else:latest=payload
        return latest

    def close(self):
        if self.process.is_alive():
            replace_pending(self.incoming,None);self.process.join(timeout=.2)
            if self.process.is_alive():self.process.terminate();self.process.join(timeout=.2)
        self.incoming.close();self.outgoing.close()
