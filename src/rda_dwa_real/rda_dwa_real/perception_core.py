"""Vectorized lidar parsing, adaptive ground removal, and supported clusters.

Coordinates are sensor -> base_link, then heights above the observed ground.
Shape labels remain geometric hints, not semantic pedestrian classifications.
"""
from dataclasses import dataclass
import math
import time
import numpy as np
from .obstacle_fusion import infer_type


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
    cluster_min_extent: float=.15
    max_cluster_points: int=8000
    structural_length: float=3.5
    structure_tile_size: float=1.2


class LidarPerceptionCore:
    def __init__(self,params=None):
        self.p=params or PerceptionParams()
        p=self.p
        positive=(p.voxel_size,p.cluster_gate,p.max_range,p.structure_tile_size,p.structural_length)
        if not all(math.isfinite(v) and v>0 for v in positive) or math.sqrt(2.)*p.voxel_size>p.cluster_gate or p.height_min>=p.height_max or p.cluster_min_points<1 or p.ground_min_points<3:
            raise ValueError('invalid perception geometry or support parameters')
        self.plane=None;self.plane_at=-math.inf


    def _ground(self,pts,stamp):
        p=self.p;r=np.linalg.norm(pts[:,:2],axis=1)
        roi=pts[(r>=p.ground_roi_min)&(r<=p.ground_roi_max)&(pts[:,2]>=-1.6)&(pts[:,2]<=.2)]
        fit=None;count=0;residual=None
        if len(roi)>=p.ground_min_points:
            hist,edges=np.histogram(roi[:,2],bins=np.arange(-1.6,.251,.05))
            k=int(np.argmax(hist));mode=(edges[k]+edges[k+1])/2
            seed=roi[np.abs(roi[:,2]-mode)<.20]
            if len(seed)>=p.ground_min_points and np.ptp(seed[:,0])>=.6 and np.ptp(seed[:,1])>=.6:
                mask=np.ones(len(seed),dtype=bool);rank=0
                for _ in range(4):
                    if np.count_nonzero(mask)<p.ground_min_points:break
                    values=seed[mask];mean=values.mean(axis=0);d=values-mean
                    xx=float(np.mean(d[:,0]**2));yy=float(np.mean(d[:,1]**2));xy=float(np.mean(d[:,0]*d[:,1]))
                    xz=float(np.mean(d[:,0]*d[:,2]));yz=float(np.mean(d[:,1]*d[:,2]));det=xx*yy-xy*xy
                    if det<=max(1e-10,1e-5*(xx+yy)**2):break
                    a=(xz*yy-yz*xy)/det;b=(yz*xx-xz*xy)/det
                    coeff=np.array([a,b,mean[2]-a*mean[0]-b*mean[1]]);rank=3
                    error=seed[:,2]-(a*seed[:,0]+b*seed[:,1]+coeff[2]);median=np.median(error[mask])
                    mad=np.median(np.abs(error[mask]-median))
                    limit=max(.025,min(.08,3.*1.4826*mad));mask=np.abs(error-median)<=limit
                if np.count_nonzero(mask)>=p.ground_min_points and rank==3 and np.isfinite(coeff).all() and math.hypot(coeff[0],coeff[1])<=math.tan(math.radians(p.ground_max_tilt_deg)):
                    fit=coeff;count=int(np.count_nonzero(mask));residual=float(np.median(np.abs(error[mask])))
        if fit is not None:
            # A large plane change is re-observed directly; small noise is smoothed.
            self.plane=fit if self.plane is None or abs(fit[2]-self.plane[2])>.15 else .7*self.plane+.3*fit
            self.plane_at=stamp
        valid=self.plane is not None and stamp-self.plane_at<=p.ground_reuse_s
        return valid,dict(ground_plane=self.plane.tolist() if self.plane is not None else None,
                          ground_inliers=count,ground_residual_m=residual,ground_age_s=stamp-self.plane_at if self.plane is not None else None)

    def process(self,xyz,stamp,pose=(0.,0.,0.)):
        begin=time.monotonic();p=self.p;points=np.asarray(xyz,dtype=float)
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
        metrics=dict(finite_points=int(np.count_nonzero(finite)),roi_points=len(points))
        if p.ground_enabled:
            phase=time.monotonic();valid,ground=self._ground(points,stamp);metrics.update(ground,ground_duration_s=time.monotonic()-phase)
            if not valid:return [],dict(metrics,valid=False,reason='ground_unavailable')
            heights=points[:,2]-(points[:,0]*self.plane[0]+points[:,1]*self.plane[1]+self.plane[2])
            keep=(heights>=p.height_min)&(heights<=p.height_max)
            metrics['ground_removed_points']=int(np.count_nonzero(heights<p.height_min))
        else:
            heights=points[:,2];keep=(heights>=p.z_min)&(heights<=p.z_max)
        points=points[keep];heights=heights[keep];metrics['filtered_points']=len(points)
        if not len(points):return [],dict(metrics,valid=True,reason='empty_observations',cluster_count=0)
        phase=time.monotonic()
        keys=np.floor(points[:,:2]/p.voxel_size).astype(np.int32)
        cells,inverse,counts=cell_index(keys)
        starts=np.r_[0,np.cumsum(counts)[:-1]];ends=np.cumsum(counts)-1
        extrema=[]
        for axis in (0,1):
            order=np.lexsort((points[:,axis],inverse));extrema.extend((order[starts],order[ends]))
        representatives=np.stack([points[idx,:2] for idx in extrema],axis=1)
        selected_count=len(np.unique(np.concatenate(extrema)))
        metrics.update(cluster_input_points=selected_count,occupied_voxels=len(cells),voxel_duration_s=time.monotonic()-phase)
        if selected_count>p.max_cluster_points:return [],dict(metrics,valid=False,reason='cluster_capacity')
        phase=time.monotonic();labels,edges=voxel_components(cells,representatives,p.voxel_size,p.cluster_gate)
        raw_labels=labels[inverse];order=np.argsort(raw_labels,kind='stable')
        cuts=np.flatnonzero(np.diff(raw_labels[order]))+1;groups=np.split(order,cuts)
        metrics.update(cluster_count=len(groups),component_edges=edges,component_duration_s=time.monotonic()-phase)
        phase=time.monotonic();detections=[];rejected=tiles=0;supports=[]
        pc,ps=math.cos(pose[2]),math.sin(pose[2])
        for indices in groups:
            raw_xy=points[indices,:2];support=len(indices);height=float(max(heights[indices]))
            cx,cy,L,W,heading=bounding_box_xy(raw_xy)
            if support<p.cluster_min_points or max(L,W)<p.cluster_min_extent-1e-8:
                rejected+=1;continue
            box_groups=[(raw_xy,(cx,cy,L,W,heading))]
            if max(L,W)>p.structural_length:
                # Group raw supports by fixed world cells in one native sort.
                world=np.column_stack((pose[0]+pc*raw_xy[:,0]-ps*raw_xy[:,1],pose[1]+ps*raw_xy[:,0]+pc*raw_xy[:,1]))
                _,ids,_=cell_index(np.floor(world/p.structure_tile_size).astype(np.int32))
                tile_order=np.argsort(ids,kind='stable');splits=np.flatnonzero(np.diff(ids[tile_order]))+1
                box_groups=[(raw_xy[idx],bounding_box_xy(raw_xy[idx])) for idx in np.split(tile_order,splits)];tiles+=len(box_groups)
                strong=[];weak=[]
                for group,box in box_groups:
                    _,_,gl,gw,_=box
                    (strong if len(group)>=p.cluster_min_points and max(gl,gw)>=p.cluster_min_extent-1e-8 else weak).append((group,box))
                for group,_ in weak:
                    distances=[float(np.min((group[:,None,0]-g[None,:,0])**2+(group[:,None,1]-g[None,:,1])**2)) for g,_ in strong]
                    if distances and min(distances)<=p.cluster_gate**2:
                        k=int(np.argmin(distances));joined=np.concatenate((strong[k][0],group));strong[k]=(joined,None)
                    else:rejected+=1
                box_groups=strong
            for group,box in box_groups:
                cx,cy,L,W,heading=box if box is not None else bounding_box_xy(group)
                if len(group)<p.cluster_min_points or max(L,W)<p.cluster_min_extent-1e-8:
                    rejected+=1;continue
                detections.append((cx,cy,infer_type(L,W),max(L,.10),max(W,.10),heading))
                supports.append(dict(points=len(group),height_m=height,structural=len(box_groups)>1))
        metrics.update(box_duration_s=time.monotonic()-phase,processing_duration_s=time.monotonic()-begin)
        metrics.update(weak_clusters_removed=rejected,structural_tiles=tiles,detection_support=supports,detection_count=len(detections))
        return detections,dict(metrics,valid=True,reason='published')


def bounding_box_xy(points):
    """Fit an OBB to every raw support point, not only voxel representatives."""
    xy=np.asarray(points,dtype=float)
    centred=xy-xy.mean(axis=0)
    xx=float(np.mean(centred[:,0]**2));yy=float(np.mean(centred[:,1]**2));cross=float(np.mean(centred[:,0]*centred[:,1]))
    theta=.5*math.atan2(2.*cross,xx-yy);c,s=math.cos(theta),math.sin(theta)
    u=c*xy[:,0]+s*xy[:,1];v=-s*xy[:,0]+c*xy[:,1]
    ulo,uhi=float(u.min()),float(u.max());vlo,vhi=float(v.min()),float(v.max())
    um,vm=(ulo+uhi)/2,(vlo+vhi)/2
    return c*um-s*vm,s*um+c*vm,max(.05,uhi-ulo),max(.05,vhi-vlo),theta


def voxel_components(cells,representatives,voxel_size,gate):
    """Exact connectivity of the four extrema, grouped by occupied XY voxel.

    Within a voxel all extrema are connected (diagonal <= gate). Pairwise
    distances across nearby occupied voxels run in bounded native arrays,
    instead of Python loops over every point and every dense neighbour.
    Raw support points are never dropped from the subsequent fitted boxes.
    """
    n=len(cells);parent=np.arange(n);left=[];right=[];edge_count=0
    if not n:return parent,0
    origin=cells.min(axis=0);span=cells.max(axis=0)-origin+1
    code=(cells[:,0].astype(np.int64)-origin[0])*int(span[1])+cells[:,1]-origin[1]
    extent=int(math.ceil(gate/voxel_size))+1;gate2=gate*gate
    for dx in range(extent+1):
        for dy in range(-extent,extent+1):
            if dx==0 and dy<=0:continue
            if (max(abs(dx)-1,0)*voxel_size)**2+(max(abs(dy)-1,0)*voxel_size)**2>gate2+1e-12:continue
            target=cells+np.array([dx,dy]);inside=(target[:,1]>=origin[1])&(target[:,1]<origin[1]+span[1])
            wanted=(target[:,0].astype(np.int64)-origin[0])*int(span[1])+target[:,1]-origin[1]
            j=np.searchsorted(code,wanted);i=np.flatnonzero(inside&(j<n))
            if not len(i):continue
            j=j[i];keep=code[j]==wanted[i];i=i[keep];j=j[keep]
            if not len(i):continue
            a=representatives[i];b=representatives[j]
            distance=(a[:,:,None,0]-b[:,None,:,0])**2+(a[:,:,None,1]-b[:,None,:,1])**2
            linked=np.min(distance,axis=(1,2))<=gate2+1e-12;i=i[linked];j=j[linked];edge_count+=len(i)
            left.append(i);right.append(j)
    if edge_count:
        left=np.concatenate(left);right=np.concatenate(right)
        while True:
            a=parent[left];b=parent[right];updated=parent.copy()
            np.minimum.at(updated,np.maximum(a,b),np.minimum(a,b))
            while np.any(updated!=updated[updated]):updated=updated[updated]
            if np.array_equal(updated,parent):break
            parent=updated
    return parent,edge_count


def cell_index(keys):
    """Native integer sort instead of structured two-column unique sorting."""
    origin=keys.min(axis=0);stride=int(keys[:,1].max()-origin[1]+1)
    code=(keys[:,0].astype(np.int64)-origin[0])*stride+keys[:,1]-origin[1]
    unique,inverse,counts=np.unique(code,return_inverse=True,return_counts=True)
    cells=np.column_stack((unique//stride+origin[0],unique%stride+origin[1])).astype(np.int32)
    return cells,inverse,counts
