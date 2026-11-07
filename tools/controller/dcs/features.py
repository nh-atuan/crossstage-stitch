"""Strict stage-availability ABI, shared with native dcs_input (48 slots)."""
import numpy as np
from tools.controller.v5.schema import PRE,POST
VERSION='dcs-pre-48-v1'
INTERACTIONS=['gradient_product','gradient_abs_difference','ncc_overlap_product','ncc_gap_overlap_product',
 'scale_k_product','scale_r_product','k_r_product','gradient_a_scale','gradient_b_scale']
NAMES=PRE+INTERACTIONS

def inputs(pre):
    x=np.asarray(pre,dtype=np.float32)
    if x.shape[-1]!=39 or not np.isfinite(x).all():raise ValueError('DCS requires finite 39-slot pre observations')
    a,b=x[...,0],x[...,1];s,k,r=x[...,30],x[...,31],x[...,32]
    extra=np.stack([a*b,np.abs(a-b),x[...,33]*x[...,38],x[...,34]*x[...,38],s*k,s*r,k*r,a*s,b*s],axis=-1)
    return np.concatenate([x,extra],axis=-1)

def columns(name):
    if name=='F0':return list(range(33))
    if name=='F1':return list(range(39))
    if name=='F2':return list(range(48))
    if name=='F0_NO_GRADIENT':return [i for i in range(33) if i not in [0,1,*range(4,26,3)]]
    if name=='F0_NO_TEXTURE':return [0,1,*range(4,26,3),30,31,32]
    raise ValueError('unknown DCS feature set')

def catalog():
    rows=[dict(index=i,name=n,availability='PRE_INITIAL',group='NCC' if 33<=i<39 else 'INTERACTION' if i>=39 else 'CONFIG' if 30<=i<33 else 'TEXTURE_GRADIENT') for i,n in enumerate(NAMES)]
    for i,n in enumerate(POST[33:],33):rows.append(dict(index=i,name=n,availability='POST_OBSERVED_ATTEMPT_ONLY',group='SPATIAL' if i>=77 else 'MATCHING_GEOMETRY_PHOTOMETRIC'))
    return rows
