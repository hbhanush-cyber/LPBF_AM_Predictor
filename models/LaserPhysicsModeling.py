import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib

matplotlib.use("Agg")

import sys,time,traceback,logging
from pathlib import Path
import numpy as np
import torch

# ================================================================
# CONFIG
# ================================================================

LINE_DATA_PATH = "layers525-650ScanLines.pt"
OUTPUT_PATH = Path("layers525-650_thermal_maps.pt")

START_LAYER = 525
END_LAYER = 650

GRID_SIZE = 2844
UNIT_SCALE = 1e-3
BUILD_EXTENT_M = 0.245
GRID_ORIGIN_M = (0.0,0.0)
PIXEL_PITCH_M = BUILD_EXTENT_M/GRID_SIZE

POWER_W = 370.0
SPEED_M_S = 800e-3
RHO = 7990.0
CP = 500.0
K_COND = 15.0
ALPHA = K_COND/(RHO*CP)
BETA_ABSORPTIVITY = 0.35
T_AMBIENT = 300.0

DEFAULT_BEAM_RADIUS_M = 50e-6
MIN_STAMP_SPACING_RADII = 2.0
JUMP_GAP_RADII = 5.0
TEMPLATE_TRAILING_RADII = 10.0
TEMPLATE_LATERAL_RADII = 6.0
TEMPLATE_RESOLUTION_FACTOR = 4
TAU_MAX_S = 5e-3
MAX_STAMPS_SAFETY_CAP = 2_000_000

# ================================================================
# LOGGING
# ================================================================

logging.basicConfig(level=logging.INFO,format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler("multi_layer_run.log",mode="w"),logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ================================================================
# THERMAL MODEL
# ================================================================

def build_template(power,speed,radius,pixel_pitch):
    sigma2 = 2.0*radius**2
    dx = pixel_pitch/TEMPLATE_RESOLUTION_FACTOR
    x_extent = TEMPLATE_TRAILING_RADII*radius
    y_extent = TEMPLATE_LATERAL_RADII*radius
    nx = max(int(2*x_extent/dx),3)
    ny = max(int(2*y_extent/dx),3)
    x_coords = np.linspace(-x_extent,x_extent,nx)
    y_coords = np.linspace(-y_extent,y_extent,ny)
    XP,YP = np.meshgrid(x_coords,y_coords,indexing="xy")
    tau = np.geomspace(1e-8,TAU_MAX_S,400)
    prefactor = 2.0*BETA_ABSORPTIVITY*power/(np.pi*RHO*CP)
    template = np.zeros_like(XP)
    denom = 4.0*ALPHA*tau+sigma2
    kernel_norm = 1.0/(np.sqrt(np.pi*ALPHA*tau)*denom)
    for i,t in enumerate(tau):
        arg = -((XP+speed*t)**2+YP**2)/denom[i]
        template += kernel_norm[i]*np.exp(arg)*(tau[min(i+1,len(tau)-1)]-tau[max(i-1,0)])*0.5
    return (template*prefactor).astype(np.float32),x_coords.astype(np.float32),y_coords.astype(np.float32)

def cooling_rate_template(template,x_coords,speed):
    return (-speed*np.gradient(template,x_coords,axis=1)).astype(np.float32)

# ================================================================
# SCAN PATH
# ================================================================

def load_layer_points(arr):
    # Raw order: x_start,x_end,y_start,y_end
    return arr[:,0],arr[:,2],arr[:,1],arr[:,3]

def downsample_layer(arr,radius):
    x1,y1,x2,y2 = load_layer_points(arr)
    if len(arr)==0:return np.array([]),np.array([]),np.array([])
    spacing = MIN_STAMP_SPACING_RADII*radius
    jump_thresh = JUMP_GAP_RADII*radius
    gap = np.hypot(x1[1:]-x2[:-1],y1[1:]-y2[:-1])
    split_idx = np.where(gap>jump_thresh)[0]+1
    groups = np.split(np.arange(len(arr)),split_idx)
    all_x,all_y,all_heading = [],[],[]

    for grp in groups:
        if len(grp)==0:continue
        xs = np.concatenate([x1[grp],x2[grp[-1:]]])
        ys = np.concatenate([y1[grp],y2[grp[-1:]]])
        seg_len = np.hypot(np.diff(xs),np.diff(ys))
        cum = np.concatenate([[0.0],np.cumsum(seg_len)])
        total_len = cum[-1]
        if total_len<1e-9:continue
        n = max(int(total_len/spacing),2)
        d = np.linspace(0,total_len,n)
        sx = np.interp(d,cum,xs)
        sy = np.interp(d,cum,ys)
        heading = np.arctan2(np.gradient(sy),np.gradient(sx)) if len(sx)>1 else np.array([0.0])
        all_x.append(sx)
        all_y.append(sy)
        all_heading.append(heading)

    if not all_x:return np.array([]),np.array([]),np.array([])
    return np.concatenate(all_x).astype(np.float32),np.concatenate(all_y).astype(np.float32),np.concatenate(all_heading).astype(np.float32)

# ================================================================
# INTERPOLATION
# ================================================================

def bilinear_sample(field,xc,yc,XP,YP):
    nx,ny = len(xc),len(yc)
    fx = (XP-xc[0])/(xc[-1]-xc[0])*(nx-1)
    fy = (YP-yc[0])/(yc[-1]-yc[0])*(ny-1)
    valid = (fx>=0)&(fx<=nx-1)&(fy>=0)&(fy<=ny-1)
    fx = np.clip(fx,0,nx-1-1e-6)
    fy = np.clip(fy,0,ny-1-1e-6)
    x0,y0 = fx.astype(int),fy.astype(int)
    x1,y1 = np.clip(x0+1,0,nx-1),np.clip(y0+1,0,ny-1)
    wx,wy = fx-x0,fy-y0
    out = field[y0,x0]*(1-wx)*(1-wy)+field[y0,x1]*wx*(1-wy)+field[y1,x0]*(1-wx)*wy+field[y1,x1]*wx*wy
    return np.where(valid,out,0.0).astype(np.float32)

# ================================================================
# COORDINATE CONVERSION
# ================================================================

def world_to_pixel(x,y):
    ox,oy = GRID_ORIGIN_M
    return (x-ox)/PIXEL_PITCH_M,(y-oy)/PIXEL_PITCH_M

# ================================================================
# STAMP ONE LAYER
# ================================================================

def stamp_layer(arr,max_map,sum_map,count_map,peak_time_map,cool_map):
    template,xc,yc = build_template(POWER_W,SPEED_M_S,DEFAULT_BEAM_RADIUS_M,PIXEL_PITCH_M)
    cool_template = cooling_rate_template(template,xc,SPEED_M_S)
    sx,sy,heading = downsample_layer(arr,DEFAULT_BEAM_RADIUS_M)
    n_points = len(sx)

    log.info(f"  {len(arr)} scan rows -> {n_points} stamp points")

    if n_points==0:return
    if n_points>MAX_STAMPS_SAFETY_CAP:raise RuntimeError(f"Safety cap exceeded: {n_points}")

    step_len = np.hypot(sx[1]-sx[0],sy[1]-sy[0]) if n_points>1 else 0.0
    dt_per_step = step_len/SPEED_M_S if SPEED_M_S>0 else 0.0

    bound_m = np.hypot(xc.max(),yc.max())
    bound_px = int(np.ceil(bound_m/PIXEL_PITCH_M))+2
    offsets = np.arange(-bound_px,bound_px+1)
    OX,OY = np.meshgrid(offsets,offsets,indexing="xy")
    GX0 = (OX*PIXEL_PITCH_M).astype(np.float32)
    GY0 = (OY*PIXEL_PITCH_M).astype(np.float32)

    H,W = max_map.shape
    spx,spy = world_to_pixel(sx,sy)
    cos_h,sin_h = np.cos(heading),np.sin(heading)
    clock = 0.0
    progress = max(n_points//10,1)

    for k in range(n_points):
        if k%progress==0:log.info(f"  stamping {k}/{n_points} ({100*k/n_points:.0f}%)")

        cx,cy = int(round(spx[k])),int(round(spy[k]))
        subx,suby = spx[k]-cx,spy[k]-cy

        i0,i1 = cx-bound_px,cx+bound_px+1
        j0,j1 = cy-bound_px,cy+bound_px+1
        ci0,ci1 = max(i0,0),min(i1,W)
        cj0,cj1 = max(j0,0),min(j1,H)

        if ci0>=ci1 or cj0>=cj1:
            clock += dt_per_step
            continue

        gx = GX0[cj0-j0:cj1-j0,ci0-i0:ci1-i0]-subx*PIXEL_PITCH_M
        gy = GY0[cj0-j0:cj1-j0,ci0-i0:ci1-i0]-suby*PIXEL_PITCH_M

        XP = gx*cos_h[k]+gy*sin_h[k]
        YP = -gx*sin_h[k]+gy*cos_h[k]

        T_patch = bilinear_sample(template,xc,yc,XP,YP)+T_AMBIENT
        cool_patch = bilinear_sample(cool_template,xc,yc,XP,YP)

        sub_max = max_map[cj0:cj1,ci0:ci1]
        is_new_max = T_patch>sub_max

        np.copyto(sub_max,T_patch,where=is_new_max)
        sum_map[cj0:cj1,ci0:ci1] += T_patch
        count_map[cj0:cj1,ci0:ci1] += 1

        peak_time_map[cj0:cj1,ci0:ci1] = np.where(
            is_new_max,clock,peak_time_map[cj0:cj1,ci0:ci1]
        )

        cool_map[cj0:cj1,ci0:ci1] = np.where(
            is_new_max,cool_patch,cool_map[cj0:cj1,ci0:ci1]
        )

        clock += dt_per_step

# ================================================================
# PROCESS ONE LAYER
# ================================================================

def process_layer(layer_data):
    if isinstance(layer_data,torch.Tensor):
        arr = layer_data.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(layer_data,dtype=np.float32)

    arr = arr[:,:4].copy()*UNIT_SCALE

    shape = (GRID_SIZE,GRID_SIZE)
    max_map = np.full(shape,T_AMBIENT,dtype=np.float32)
    sum_map = np.zeros(shape,dtype=np.float32)
    count_map = np.zeros(shape,dtype=np.float32)
    peak_time_map = np.zeros(shape,dtype=np.float32)
    cool_map = np.zeros(shape,dtype=np.float32)

    stamp_layer(arr,max_map,sum_map,count_map,peak_time_map,cool_map)

    avg_map = np.where(
        count_map>0,
        sum_map/np.clip(count_map,1,None),
        T_AMBIENT
    ).astype(np.float32)

    gy,gx = np.gradient(max_map,PIXEL_PITCH_M)
    grad_map = np.hypot(gx,gy).astype(np.float32)

    # [4,2844,2844]
    # 0 = max temp
    # 1 = average temp
    # 2 = cooling rate
    # 3 = temperature gradient
    return np.stack([max_map,avg_map,cool_map,grad_map],axis=0)

# ================================================================
# MAIN
# ================================================================

def main():
    log.info("="*60)
    log.info(f"Processing layers {START_LAYER}-{END_LAYER}")
    log.info(f"Output: {OUTPUT_PATH.resolve()}")
    log.info("="*60)

    raw = torch.load(LINE_DATA_PATH,map_location="cpu",weights_only=False)
    log.info(f"Loaded {len(raw)} layer keys")

    layer_map = {}
    for key in raw.keys():
        try:layer_map[int(key)] = key
        except (ValueError,TypeError):pass

    requested_layers = list(range(START_LAYER,END_LAYER+1))
    available_layers = [x for x in requested_layers if x in layer_map]
    missing_layers = [x for x in requested_layers if x not in layer_map]

    log.info(f"Found {len(available_layers)}/{len(requested_layers)} requested layers")

    if missing_layers:
        log.warning(f"Missing layers: {missing_layers}")

    all_maps = []
    processed_layers = []

    total_start = time.time()

    for i,layer_num in enumerate(available_layers,1):
        log.info("="*60)
        log.info(f"LAYER {layer_num} ({i}/{len(available_layers)})")
        log.info("="*60)

        try:
            maps = process_layer(raw[layer_map[layer_num]])
            all_maps.append(torch.from_numpy(maps))
            processed_layers.append(layer_num)

            log.info(f"Layer {layer_num} complete: {maps.shape}")

        except Exception:
            log.error(f"FAILED layer {layer_num}")
            log.error(traceback.format_exc())

    if not all_maps:
        raise RuntimeError("No layers were successfully processed.")

    # [number_of_layers,4,2844,2844]
    all_maps = torch.stack(all_maps,dim=0)

    output = {
        "maps": all_maps,
        "layers": torch.tensor(processed_layers,dtype=torch.long),
        "channels": ["max_temp","avg_temp","cooling_rate","grad_temp"],
        "grid_size": GRID_SIZE,
        "build_extent_m": BUILD_EXTENT_M,
        "pixel_pitch_m": PIXEL_PITCH_M,
        "power_W": POWER_W,
        "speed_m_s": SPEED_M_S,
        "beam_radius_m": DEFAULT_BEAM_RADIUS_M,
    }

    torch.save(output,OUTPUT_PATH)

    log.info("="*60)
    log.info("DONE")
    log.info(f"Saved: {OUTPUT_PATH.resolve()}")
    log.info(f"Final tensor shape: {all_maps.shape}")
    log.info(f"Processed layers: {processed_layers[0]}-{processed_layers[-1]}")
    log.info(f"Total time: {(time.time()-total_start)/60:.2f} minutes")
    log.info("="*60)

if __name__=="__main__":
    try:main()
    except Exception:
        log.error(traceback.format_exc())
        sys.exit(1)

