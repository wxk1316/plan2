"""Vectorized lidar parsing, adaptive ground removal, and supported clusters.

Coordinates are sensor -> base_link, then heights above the observed ground.
Shape labels remain geometric hints, not semantic pedestrian classifications.
"""
from dataclasses import dataclass
import math
import numpy as np
from .obstacle_fusion import Clusterer, cluster_bounding_box, infer_type


def point_cloud_xyz(msg):
    """Read XYZ float fields with byte order, point stride and row padding."""
    height, width = int(msg.height), int(msg.width)
    step = int(msg.point_step)
    row_step = int(getattr(msg, 'row_step', width*step))
    if height<=0 or width<=0 or step<=0 or row_step<width*step:
        raise ValueError('invalid cloud dimensions')
    fields={f.name:f for f in msg.fields}
    formats=[];offsets=[]
    for name in ('x','y','z'):
        f=fields.get(name)
        if f is None or int(getattr(f,'count',1))<1 or f.datatype not in (7,8):
            raise ValueError('XYZ float fields required')
        size=4 if f.datatype==7 else 8
        if f.offset<0 or f.offset+size>step:raise ValueError('invalid field offset')
        formats.append(('>' if getattr(msg,'is_bigendian',False) else '<')+f'f{size}')
        offsets.append(f.offset)
    if len(msg.data)<(height-1)*row_step+width*step:raise ValueError('truncated cloud')
    dtype=np.dtype(dict(names=['x','y','z'],formats=formats,offsets=offsets,itemsize=step))
    rows=np.ndarray((height,width),dtype=dtype,buffer=msg.data,strides=(row_step,step))
    return np.column_stack([rows[n].ravel() for n in ('x','y','z')]).astype(float,copy=False)


@dataclass
class PerceptionParams:
    vehicle_length: float=1.2
    vehicle_width: float=.8
    vehicle_outside: float=.05
    sensor_x: float=0.
    sensor_y: float=0.
    sensor_yaw: float=0.
    lidar_height: float=.45
    max_range: float=6.
    x_min: float=-10.
    x_max: float=10.
    y_min: float=-6.
    y_max: float=6.
    z_min: float=-.5
    z_max: float=.5
    ground_enabled: bool=True
    ground_roi_min: float=.8
    ground_roi_max: float=4.
    ground_min_points: int=50
    ground_max_tilt_deg: float=15.
    ground_reuse_s: float=1.
    height_min: float=.12
    height_max: float=2.2
    voxel_size: float=.10
    cluster_gate: float=.30
    cluster_min_points: int=10
    cluster_min_extent: float=.20
    max_cluster_points: int=8000
    structural_length: float=3.5
    structure_tile_size: float=1.2


class LidarPerceptionCore:
    def __init__(self,params=None):
        self.p=params or PerceptionParams()
        p=self.p
        positive=(p.voxel_size,p.cluster_gate,p.max_range,p.structure_tile_size,p.structural_length)
        if not all(math.isfinite(v) and v>0 for v in positive) or p.height_min>=p.height_max or p.cluster_min_points<1 or p.ground_min_points<3:
            raise ValueError('invalid perception geometry or support parameters')
        self.plane=None;self.plane_at=-math.inf
        self.clusterer=Clusterer(self.p.cluster_gate,1,self.p.max_range+1.)

    def _ground(self,pts,stamp):
        p=self.p;r=np.linalg.norm(pts[:,:2],axis=1)
        roi=pts[(r>=p.ground_roi_min)&(r<=p.ground_roi_max)&(pts[:,2]>=-1.6)&(pts[:,2]<=.2)]
        fit=None;count=0;residual=None
        if len(roi)>=p.ground_min_points:
            hist,edges=np.histogram(roi[:,2],bins=np.arange(-1.6,.251,.05))
            k=int(np.argmax(hist));mode=(edges[k]+edges[k+1])/2
            seed=roi[np.abs(roi[:,2]-mode)<.20]
            if len(seed)>=p.ground_min_points and np.ptp(seed[:,0])>=.6 and np.ptp(seed[:,1])>=.6:
                a=np.column_stack((seed[:,:2],np.ones(len(seed))));mask=np.ones(len(seed),dtype=bool)
                for _ in range(4):
                    if sum(mask)<p.ground_min_points:break
                    coeff,_,rank,_=np.linalg.lstsq(a[mask],seed[mask,2],rcond=None)
                    error=seed[:,2]-a@coeff;median=np.median(error[mask])
                    mad=np.median(np.abs(error[mask]-median))
                    limit=max(.025,min(.08,3.*1.4826*mad))
                    mask=np.abs(error-median)<=limit
                if sum(mask)>=p.ground_min_points and rank==3 and np.isfinite(coeff).all() and math.hypot(coeff[0],coeff[1])<=math.tan(math.radians(p.ground_max_tilt_deg)):
                    fit=coeff;count=int(sum(mask));residual=float(np.median(np.abs(error[mask])))
        if fit is not None:
            # A large plane change is re-observed directly; small noise is smoothed.
            self.plane=fit if self.plane is None or abs(fit[2]-self.plane[2])>.15 else .7*self.plane+.3*fit
            self.plane_at=stamp
        valid=self.plane is not None and stamp-self.plane_at<=p.ground_reuse_s
        return valid,dict(ground_plane=self.plane.tolist() if self.plane is not None else None,
                          ground_inliers=count,ground_residual_m=residual,ground_age_s=stamp-self.plane_at if self.plane is not None else None)

    def process(self,xyz,stamp,pose=(0.,0.,0.)):
        p=self.p;points=np.asarray(xyz,dtype=float)
        if points.ndim!=2 or points.shape[1]!=3:raise ValueError('XYZ array required')
        finite=np.isfinite(points).all(axis=1);points=points[finite].copy()
        c,s=math.cos(p.sensor_yaw),math.sin(p.sensor_yaw)
        x=points[:,0].copy();y=points[:,1].copy()
        points[:,0]=c*x-s*y+p.sensor_x;points[:,1]=s*x+c*y+p.sensor_y
        points[:,2]+=p.lidar_height  # TF base_link -> rslidar has +height
        radius=np.linalg.norm(points[:,:2],axis=1)
        keep=(radius<=p.max_range)&(points[:,0]>=p.x_min)&(points[:,0]<=p.x_max)&(points[:,1]>=p.y_min)&(points[:,1]<=p.y_max)
        self_box=(np.abs(points[:,0])<=p.vehicle_length/2+p.vehicle_outside)&(np.abs(points[:,1])<=p.vehicle_width/2+p.vehicle_outside)
        points=points[keep&~self_box]
        metrics=dict(finite_points=int(sum(finite)),roi_points=len(points))
        if p.ground_enabled:
            valid,ground=self._ground(points,stamp);metrics.update(ground)
            if not valid:return [],dict(metrics,valid=False,reason='ground_unavailable')
            heights=points[:,2]-(points[:,:2]@self.plane[:2]+self.plane[2])
            keep=(heights>=p.height_min)&(heights<=p.height_max)
            metrics['ground_removed_points']=int(sum(heights<p.height_min))
        else:
            heights=points[:,2];keep=(heights>=p.z_min)&(heights<=p.z_max)
        points=points[keep];heights=heights[keep];metrics['filtered_points']=len(points)
        if not len(points):return [],dict(metrics,valid=True,reason='empty_observations',cluster_count=0)
        keys=np.floor(points[:,:2]/p.voxel_size).astype(np.int32)
        _,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
        starts=np.r_[0,np.cumsum(counts)[:-1]];ends=np.cumsum(counts)-1
        extrema=[]
        for axis in (0,1):
            order=np.lexsort((points[:,axis],inverse))
            extrema.extend((order[starts],order[ends]))
        selected_indices=np.unique(np.concatenate(extrema))
        selected=[tuple(q) for q in points[selected_indices,:2]]
        point_cell={tuple(points[i,:2]):int(inverse[i]) for i in selected_indices}
        metrics['cluster_input_points']=len(selected)
        if len(selected)>p.max_cluster_points:return [],dict(metrics,valid=False,reason='cluster_capacity')
        clusters=self.clusterer.cluster(selected);metrics['cluster_count']=len(clusters)
        detections=[];rejected=tiles=0;supports=[]
        pc,ps=math.cos(pose[2]),math.sin(pose[2])
        for cluster in clusters:
            ids=list({point_cell[q] for q in cluster})
            raw_mask=np.isin(inverse,ids);raw_xy=points[raw_mask,:2]
            support=int(sum(raw_mask));height=float(max(heights[raw_mask]))
            cx,cy,L,W,heading=bounding_box_xy(raw_xy)
            if support<p.cluster_min_points or max(L,W)<p.cluster_min_extent-1e-8:
                rejected+=1;continue
            groups=[raw_xy]
            if max(L,W)>p.structural_length:
                # Fixed world cells prevent a long facade's box filling a courtyard.
                grid={}
                for x,y in raw_xy:
                    wx=pose[0]+pc*x-ps*y;wy=pose[1]+ps*x+pc*y
                    k=(math.floor(wx/p.structure_tile_size),math.floor(wy/p.structure_tile_size))
                    grid.setdefault(k,[]).append((x,y))
                groups=list(grid.values());tiles+=len(groups)
                strong=[];weak=[]
                for group in groups:
                    _,_,gl,gw,_=bounding_box_xy(group)
                    (strong if len(group)>=p.cluster_min_points and max(gl,gw)>=p.cluster_min_extent-1e-8 else weak).append(group)
                # Keep contiguous facade ends by merging a short fringe into
                # its supported neighbour. Disconnected weak speckles vanish.
                for group in weak:
                    raw=np.asarray(group)
                    distances=[float(np.min(np.linalg.norm(raw[:,None,:]-np.asarray(g)[None,:,:],axis=2))) for g in strong]
                    if distances and min(distances)<=p.cluster_gate:
                        strong[int(np.argmin(distances))].extend(group)
                    else:rejected+=1
                groups=strong
            for group in groups:
                cx,cy,L,W,heading=bounding_box_xy(group)
                # Apply support to each tile, not its entire parent facade. A
                # one-return fringe of a large wall is not a road barrier.
                if len(group)<p.cluster_min_points or max(L,W)<p.cluster_min_extent-1e-8:
                    rejected+=1;continue
                detections.append((cx,cy,infer_type(L,W),max(L,.10),max(W,.10),heading))
                supports.append(dict(points=len(group),height_m=height,structural=len(groups)>1))
        metrics.update(weak_clusters_removed=rejected,structural_tiles=tiles,detection_support=supports,detection_count=len(detections))
        return detections,dict(metrics,valid=True,reason='published')


def bounding_box_xy(points):
    """Fit an OBB to every raw support point, not only voxel representatives."""
    xy=np.asarray(points,dtype=float)
    centred=xy-xy.mean(axis=0);cov=centred.T@centred/max(1,len(xy))
    theta=.5*math.atan2(2.*cov[0,1],cov[0,0]-cov[1,1])
    c,s=math.cos(theta),math.sin(theta)
    rotation=np.array([[c,-s],[s,c]])
    projected=xy@rotation;lo=projected.min(axis=0);hi=projected.max(axis=0)
    centre=((lo+hi)/2)@rotation.T
    return float(centre[0]),float(centre[1]),float(max(.05,hi[0]-lo[0])),float(max(.05,hi[1]-lo[1])),theta
