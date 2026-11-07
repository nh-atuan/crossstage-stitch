"""Versioned online-only feature names; offline QoR targets are not features."""
from tools.controller.feature_schema import POOLED_PRE,POOLED_POST
VERSION='controller-v5-features-1'
SPATIAL=['stratified_ssim_min','stratified_ssim_q25','stratified_ssim_median','stratified_ssim_q75',
    'stratified_mae_q90','fully_valid_patch_fraction','quadrant_valid_0','quadrant_valid_1','quadrant_valid_2','quadrant_valid_3',
    'jacobian_abs_det_min','jacobian_abs_det_max','jacobian_det_spread','invalid_projection_fraction','stratified_ssim_mean','stratified_ssim_std']
PRE=list(POOLED_PRE)
POST=list(POOLED_POST)+SPATIAL
assert len(PRE)==39 and len(POST)==93

def mapping():return dict(version=VERSION,pre=PRE,post=POST,
    pre_availability='scene/config and optional pair NCC before candidate attempt; no match/H or Reference observations',
    post_availability='candidate S4/S5/H only; stratified patches use A/B and candidate H, never Reference H',
    invalid='Invalid H/status/payload are rejected, not imputed as successful observations',
    spatial_groups=dict(photo=list(range(77,87))+[91,92],projection=list(range(87,91))))
