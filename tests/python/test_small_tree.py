import numpy as np
import pytest
from tools.controller.small_tree import fit, predict
from tools.controller.learned_measure import same_h
from tools.controller.feature_schema import selector_features


def test_cost_tree_learns_two_features_and_boundary_matches_cpp():
    x=np.array([[0,0],[0,1],[1,0],[1,1]]*8,dtype=float)
    # Opposite actions in disjoint regions require more than one threshold.
    desired=((x[:,0]>0) | (x[:,1]>0)).astype(int)
    costs=np.stack([desired,1-desired],axis=1)
    nodes=fit(x,costs,min_leaf=4,max_depth=2)
    assert np.array_equal(predict(nodes,x),desired)
    root=nodes[0];a=np.zeros((1,2));a[0,root['feature']]=root['threshold']
    assert predict(nodes,a)[0]==0
    assert len(nodes)<=7


def test_pure_gate_cannot_accept_mixed_leaf_or_nonfinite_inputs():
    x=np.zeros((16,2));truth=np.array([True]*15+[False])
    costs=np.stack([truth,10*(~truth)],axis=1)
    nodes=fit(x,costs,labels=truth,pure_accept=True)
    assert predict(nodes,x).sum()==0
    with pytest.raises(ValueError):fit([[np.nan]],[[0,1]])
    with pytest.raises(ValueError):predict(nodes,[[np.inf,0]])


def test_failed_nan_payloads_match_but_changed_numeric_h_does_not():
    failure={f'h{i}':'nan' for i in range(9)}
    assert same_h(failure,failure)
    model={f'h{i}':str(int(i in [0,4,8])) for i in range(9)}
    assert not same_h(model,failure)
    other=dict(model,h2='0.0000000001')
    assert not same_h(model,other)
    x=np.arange(26,dtype=float)
    assert np.array_equal(selector_features(x)[26:],np.full(4,12.))
