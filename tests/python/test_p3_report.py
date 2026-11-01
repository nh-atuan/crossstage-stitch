"""Publication assets are generated from real-shaped raw CSVs, never fixture numbers in source."""
import json
import cv2
import numpy as np
import pytest

from tools.controller.p3_report import build, check_reuse_equivalence
from tools.controller.p3_cases import export as export_cases
from test_p3_evaluation import write_csv, run_row


def test_end_to_end_report_assets_and_correctness_guard(tmp_path):
    udis, hp, output = [tmp_path / p for p in ('udis','hp','output')]
    udis.mkdir();hp.mkdir()
    pair='v_test_1_2'
    rows=[run_row(repeat=i,num_matches=40) for i in range(10)]
    for mode in ('fixed','adaptive','no_reuse'):
        write_csv(udis/f'{mode}.csv',rows)
        if mode!='no_reuse':
            write_csv(hp/f'{mode}.csv',rows)
    for mode in ('adaptive','no_reuse'):
        traces=[]
        for i in range(10):
            row=run_row(repeat=i,num_matches=40)
            for attempt in range(2):
                traces.append({**row,'attempt_idx':attempt,'R':128 if attempt==0 else 256,
                               'eligible_R_old':0 if attempt==0 else 128,'generation':1,
                               'num_inliers':5 if attempt==0 else 20,'action':'RETRY' if attempt==0 else 'ACCEPT',
                               'reason':'TOO_FEW_INLIERS' if attempt==0 else 'SELF_CHECK_PASSED',
                               'homography_status':0,'t_s4':0 if mode=='adaptive' else 1,
                               't_s5':1,'t_attempt':2 if mode=='adaptive' else 3})
        write_csv(udis/f'{mode}-attempts.csv',traces)
    (udis/'campaign.json').write_text(json.dumps({'inputs':{pair:{'a':'a','b':'b'}}}))
    manifest=tmp_path/'pairs.json'
    manifest.write_text(json.dumps({pair:dict(sequence='v_test',width_b=100,height_b=80,H_ba=np.eye(3).tolist())}))
    report=build(udis,hp,manifest,output)
    assert report['reuse_equivalence']=='passed_all_pairs_all_repeats'
    assert report['controller']['retry']['pairs_with_retry']==1
    assert report['a2']['pairs']==1
    assert report['geometry_comparison']['paired_adaptive_mean_px']==0
    for name in ('controller-results.tex','controller-ladder.pdf','controller-ladder.svg','a2-retry.pdf','generation.json'):
        assert (output/name).stat().st_size>0
    image=tmp_path/'image.ppm'
    cv2.imwrite(str(image),np.zeros((20,30,3),dtype=np.uint8))
    visual_manifest=tmp_path/'visual-pairs.json'
    visual_manifest.write_text(json.dumps({pair:dict(a='image.ppm',b='image.ppm')}))
    cases=export_cases(udis,visual_manifest,output/'cases')
    assert cases['selected'][0]['recovered_inliers']==15
    assert (output/'cases/case-review.png').stat().st_size>0
    rows[0]['num_inliers']=21
    write_csv(udis/'no_reuse.csv',rows)
    with pytest.raises(ValueError,match='num_inliers'):
        check_reuse_equivalence(udis/'adaptive.csv',udis/'no_reuse.csv')
    rows[0]['num_inliers']=20
    for row in rows:row['seed']=8
    write_csv(udis/'no_reuse.csv',rows)
    with pytest.raises(ValueError,match='seed'):
        check_reuse_equivalence(udis/'adaptive.csv',udis/'no_reuse.csv')
