"""Physical 1.2 x 0.8 m rectangles and bounded, time-indexed motion checks.

All poses in this module are GEOMETRIC CENTRES. Safety clearance is separate
from physical dimensions. Circular radii are used only in conservative bounds.
"""
from dataclasses import dataclass
import math
import numpy as np
from .collision_checker import _rect_corners_vec, _rect_rect_gaps_vec

@dataclass(frozen=True)
class VehicleGeometry:
    length: float = 1.2
    width: float = .8
    wheelbase: float = .69
    @property
    def rear_offset(self): return self.wheelbase / 2.
    @property
    def radius(self): return math.hypot(self.length/2, self.width/2)

@dataclass(frozen=True)
class TrackedObstacle:
    x: float
    y: float
    length: float
    width: float
    heading: float = 0.
    vx: float = 0.
    vy: float = 0.
    ident: int = 0
    uncertain: bool = False
    uncertainty_speed: float = 0.
    @property
    def speed(self): return math.hypot(self.vx,self.vy)


def steering_step(angle,rate,target,dt,max_rate,max_acc,deadband=math.radians(.15)):
    """Discrete braking-aware steering servo without snapping its rate to zero."""
    angle=np.asarray(angle,dtype=float);rate=np.asarray(rate,dtype=float)
    error=np.asarray(target,dtype=float)-angle
    allowed=np.sqrt(2.*max_acc*np.abs(error)+(max_acc*dt)**2)-max_acc*dt
    wanted=np.sign(error)*np.minimum(max_rate,np.maximum(0.,allowed))
    settle=(np.abs(error)<=deadband)&(np.abs(rate)<=max_acc*dt)
    wanted=np.where(settle,0.,wanted)
    next_rate=rate+np.clip(wanted-rate,-max_acc*dt,max_acc*dt)
    return angle+next_rate*dt,next_rate


def speed_step(speed,acceleration,target,dt,acc,dec,jerk):
    """Braking-aware velocity servo with continuous bounded acceleration."""
    speed=np.asarray(speed,dtype=float);acceleration=np.asarray(acceleration,dtype=float)
    error=np.asarray(target,dtype=float)-speed
    allowed=np.sqrt(2.*jerk*np.abs(error)+(jerk*dt)**2)-jerk*dt
    wanted=np.sign(error)*np.minimum(np.abs(error)/.3,allowed)
    wanted=np.clip(wanted,-dec,acc)
    wanted=np.where((np.abs(error)<.0001)&(np.abs(acceleration)<jerk*dt),0.,wanted)
    next_acc=acceleration+np.clip(wanted-acceleration,-jerk*dt,jerk*dt)
    return np.maximum(0.,speed+next_acc*dt),next_acc


def obstacle_from_object(o):
    typ=str(getattr(o,'obstacle_type','static_large'))
    speed=float(getattr(o,'speed',0.) or 0.)
    direction=float(getattr(o,'heading',0.) or 0.)
    # Upstream dynamic heading is velocity direction; static footprint heading
    # is explicitly preserved when available. Do not rotate a box with velocity.
    heading=float(getattr(o,'shape_heading',getattr(o,'theta',direction)) or 0.)
    length=float(getattr(o,'length',0.) or 0.)
    width=float(getattr(o,'width',0.) or 0.)
    radius=float(getattr(o,'radius',.15) or .15)
    if length<=0 or width<=0: length=width=2*radius
    uncertain=typ=='unknown'
    if typ in ('static_large','block','static_small'):speed=0.
    return TrackedObstacle(float(getattr(o,'x',getattr(o,'fx',0.))),
                           float(getattr(o,'y',getattr(o,'fy',0.))),length,width,
                           heading,speed*math.cos(direction),speed*math.sin(direction),
                           int(getattr(o,'id',0)),uncertain,1.5 if uncertain else 0.)


def minimum_gaps(poses, times, obstacles, geometry, age=0., interval_bound=0.):
    """Conservative OBB separation at samples, minus uncertainty/sweep bounds.

    poses: (..., T, 3), times: (T,). Returns shape (...,), minimum gap over T.
    Moving obstacles are translated at EVERY trajectory time, not only at end.
    Four unit separating axes give a lower bound on Euclidean clearance. This
    avoids per-corner distance tensors and never overstates positive clearance.
    """
    poses=np.asarray(poses,dtype=float);times=np.asarray(times,dtype=float)
    shape=poses.shape[:-2]; count=int(np.prod(shape)) if shape else 1
    flat=poses.reshape((-1,3)); T=poses.shape[-2]
    bc=np.cos(flat[:,2]);bs=np.sin(flat[:,2]);hl,hw=geometry.length/2,geometry.width/2
    gaps=np.full((count,T),np.inf)
    for o in obstacles:
        if not all(math.isfinite(x) for x in (o.x,o.y,o.length,o.width,o.vx,o.vy,o.heading)):
            return np.full(shape,-np.inf)
        tt=np.maximum(0.,times+age)
        # Unknown means any direction up to a speed bound, centred at the
        # observation; translating that disk by an unconfirmed velocity can
        # miss motion in the opposite direction.
        ox=np.tile(o.x+(0. if o.uncertain else o.vx)*tt,count)
        oy=np.tile(o.y+(0. if o.uncertain else o.vy)*tt,count)
        dx=flat[:,0]-ox;dy=flat[:,1]-oy
        oc,os=math.cos(o.heading),math.sin(o.heading)
        co=np.abs(bc*oc+bs*os);si=np.abs(bs*oc-bc*os)
        ol,ow=o.length/2,o.width/2
        a=np.abs(dx*bc+dy*bs)-(hl+ol*co+ow*si)
        b=np.abs(-dx*bs+dy*bc)-(hw+ol*si+ow*co)
        c=np.abs(dx*oc+dy*os)-(ol+hl*co+hw*si)
        e=np.abs(-dx*os+dy*oc)-(ow+hl*si+hw*co)
        d=np.maximum(np.maximum(a,b),np.maximum(c,e))
        d=d.reshape(count,T)
        d-=o.uncertainty_speed*tt[None,:]
        obstacle_speed=o.uncertainty_speed if o.uncertain else o.speed
        d-=interval_bound + obstacle_speed*(times[1]-times[0])/2 if len(times)>1 else interval_bound
        gaps=np.minimum(gaps,d)
    return np.min(gaps,axis=-1).reshape(shape)


def rollout_rear(x,y,yaw,speed,steer,vs,ds,geometry,horizon=3.,dt=.1,
                 acc=.7,dec=1.2,steer_rate=math.radians(35),jerk=None,
                 steer_accel=None,initial_acc=0.,initial_steer_rate=0.,delay=0.,with_actuator_state=False):
    """Ramp-constrained bicycle rollout. Returns rear poses and actual speeds."""
    vs=np.asarray(vs,dtype=float);ds=np.asarray(ds,dtype=float);n=len(vs)
    times=np.linspace(0.,horizon,int(math.ceil(horizon/dt))+1);step=times[1]-times[0]
    rear=np.empty((n,len(times),3));vv=np.empty((n,len(times)))
    vec=lambda value:np.broadcast_to(np.asarray(value,dtype=float),(n,)).copy()
    xx=vec(x);yy=vec(y);th=vec(yaw);v=np.maximum(0.,vec(speed));delta=vec(steer)
    acceleration=vec(initial_acc);rate=vec(initial_steer_rate)
    steering=np.empty_like(vv);accelerations=np.empty_like(vv);rates=np.empty_like(vv)
    rear[:,0,:]=np.stack((xx,yy,th),axis=1);vv[:,0]=v
    steering[:,0]=delta;accelerations[:,0]=acceleration;rates[:,0]=rate
    for j in range(1,len(times)):
        target_v=vs[:,j] if vs.ndim==2 else vs
        target_d=ds[:,j] if ds.ndim==2 else ds
        if times[j]<=delay:target_v=vec(speed);target_d=vec(steer)
        if jerk is None:v+=np.clip(target_v-v,-dec*step,acc*step)
        else:
            v,acceleration=speed_step(v,acceleration,target_v,step,acc,dec,jerk)
        if steer_accel is None:delta+=np.clip(target_d-delta,-steer_rate*step,steer_rate*step)
        else:
            delta,rate=steering_step(delta,rate,target_d,step,steer_rate,steer_accel)
        omega=v*np.tan(delta)/geometry.wheelbase
        mid=th+omega*step/2
        xx+=v*np.cos(mid)*step; yy+=v*np.sin(mid)*step; th+=omega*step
        rear[:,j,:]=np.stack((xx,yy,th),axis=1);vv[:,j]=v
        steering[:,j]=delta;accelerations[:,j]=acceleration;rates[:,j]=rate
    return (rear,vv,times,steering,accelerations,rates) if with_actuator_state else (rear,vv,times)


def to_centres(rear,geometry):
    out=np.array(rear,copy=True)
    out[...,0]+=geometry.rear_offset*np.cos(out[...,2])
    out[...,1]+=geometry.rear_offset*np.sin(out[...,2])
    return out


def spin_is_safe(x,y,yaw,target,obstacles,geometry,age=0.,wait=3.,speed=.6,margin=.12):
    err=(target-yaw+math.pi)%(2*math.pi)-math.pi
    duration=wait+abs(err)/max(speed,.05)+1.
    times=np.linspace(0,duration,int(math.ceil(duration/.1))+1)
    fraction=np.clip((times-wait)/max(abs(err)/max(speed,.05),.01),0,1)
    poses=np.stack((np.full(len(times),x),np.full(len(times),y),yaw+err*fraction),axis=1)
    bound=(speed*geometry.radius)*.05
    return bool(minimum_gaps(poses,times,obstacles,geometry,age,bound)>=margin)

def rollout_stop(x,y,yaw,speed,steer,vs,ds,geometry,horizon=3.,dt=.1,reaction=.25,dec=.8,
                 jerk=None,steer_accel=None,initial_acc=0.,initial_steer_rate=0.):
    """Candidate's first action, delayed braking, then stationary occupancy.

    This guards safe stopping locations for moving obstacles: a nominal cruise
    path alone can become unsafe if tracking or the next plan brakes early.
    Braking assumption is deliberately slower than the commanded 1.2 m/s².
    """
    vs=np.asarray(vs,dtype=float);ds=np.asarray(ds,dtype=float);n=len(vs)
    tt=np.linspace(0.,horizon,int(math.ceil(horizon/dt))+1);h=tt[1]-tt[0]
    vec=lambda value:np.broadcast_to(np.asarray(value,dtype=float),(n,)).copy()
    rear=np.empty((n,len(tt),3));xx=vec(x);yy=vec(y);th=vec(yaw);vv=np.maximum(0.,vec(speed));delta=vec(steer)
    acceleration=vec(initial_acc);rate=vec(initial_steer_rate);held_target=vec(steer)
    rear[:,0]=np.column_stack((xx,yy,th))
    for j,t in enumerate(tt[1:],1):
        target=vs if t<=reaction else np.zeros(n)
        if jerk is None:vv+=np.clip(target-vv,-dec*h,.7*h)
        else:
            vv,acceleration=speed_step(vv,acceleration,target,h,.7,dec,jerk)
        if t<=reaction:
            if steer_accel is None:delta+=np.clip(ds-delta,-math.radians(30)*h,math.radians(30)*h)
            else:
                delta,rate=steering_step(delta,rate,ds,h,math.radians(30),steer_accel)
            held_target=delta.copy()
        elif steer_accel is not None:
            delta,rate=steering_step(delta,rate,held_target,h,math.radians(30),steer_accel)
        omega=vv*np.tan(delta)/geometry.wheelbase;mid=th+omega*h/2
        xx+=vv*np.cos(mid)*h;yy+=vv*np.sin(mid)*h;th+=omega*h
        rear[:,j]=np.column_stack((xx,yy,th))
    return rear,tt
