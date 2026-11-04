import importlib.util
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(importlib.util.find_spec('xgboost') is None,
    reason='optional offline training dependency: requirements-learn.txt')


def test_booster_export_matches_margins_including_float32_boundaries():
    from tools.controller import boosted_model as bm
    x=np.linspace(-2,2,160).reshape(-1,1);y=(x[:,0]>.2).astype(float)
    for classifier,target in [(True,y),(False,2*x[:,0]+3)]:
        model=bm.fit(x,target,classifier=classifier,rounds=12)
        exported=bm.export(model,transform='sigmoid' if classifier else 'exp')
        values=np.r_[x,[[np.nextafter(np.float32(.2),np.float32(1.))]]]
        assert np.allclose(bm.predict(exported,values,raw=True),bm.margins(model,values),atol=2e-6)
        with pytest.raises(ValueError,match='invalid exported'):bm.predict(exported,[[np.nan]])


def test_probability_calibration_is_finite_for_extreme_and_constant_labels():
    from tools.controller import boosted_model as bm
    for y in [np.zeros(40),np.ones(40),np.arange(40)%2]:
        result=bm.calibrate(np.linspace(-5,5,40),y)
        assert np.isfinite(list(result.values())).all()
        assert result['slope']>=0


def test_group_folds_keep_every_configuration_of_a_component_together():
    from tools.controller.train_pooled import folds
    groups=np.array(['a','a','b','c','d','e','f','b'])
    outer=folds(groups)
    assert outer[0]==outer[1] and outer[2]==outer[-1]
    assert np.array_equal(outer,folds(groups))
    for f in set(outer):assert not set(groups[outer==f])&set(groups[outer!=f])


def test_expected_cost_can_select_k500_without_per_configuration_gate_filter():
    from tools.controller.pooled_policy import choose
    p=np.array([[.95,.99,1.],[.1,.99,1.]])
    latency=np.array([[2.,7.,10.],[2.,7.,10.]])
    assert choose(p,latency,2,.8,.5).tolist()==[0,1]
    assert choose(p,latency,2,1.,.5).tolist()==[2,2]


def test_replay_counts_false_accept_and_charges_rejected_attempt_before_fallback():
    from tools.controller.pooled_data import Grid
    from tools.controller.calibrate import REFERENCE
    from tools.controller.pooled_policy import replay,tune
    grid=Grid(['a','b'],['cheap',REFERENCE],[],np.zeros((2,2,1)),np.zeros((2,2,1)),
        np.array([[1,1],[0,1]],bool),np.ones((2,2),bool),np.array([[2.,10.],[2.,10.]]),
        np.ones((2,2)),np.full((2,2),.1),np.array(['a','b']))
    prediction=(np.array([[.99,1.],[.99,1.]]),grid.latency,np.array([[.99,1.],[.8,1.]]))
    loose=replay(grid,prediction,pre_threshold=0.,gate_threshold=.5)
    assert loose['false_accept']==1 and loose['cheap_stops']==2
    strict=replay(grid,prediction,pre_threshold=0.,gate_threshold=.9)
    assert strict['false_accept']==0 and strict['retry']==1
    assert strict['cost'][1]==pytest.approx(1.+.1+10.+.5)
    _,selected=tune(grid,prediction)
    assert selected['false_accept']==0
