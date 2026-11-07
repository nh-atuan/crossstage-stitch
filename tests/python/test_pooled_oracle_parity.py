import pytest
from tools.controller.holdout_oracle import verify_successful_policy


def test_successful_policy_must_match_fixed_configuration_and_homography():
    row=dict(config_id='candidate',status='OK',fail_stage='',num_matches='10',num_inliers='8',seed='20261101',
        **{f'h{i}':str(int(i in [0,4,8])) for i in range(9)})
    verify_successful_policy(row,dict(row))
    with pytest.raises(ValueError,match='homography'):verify_successful_policy(dict(row,h2='1'),row)
    with pytest.raises(ValueError,match='configuration'):verify_successful_policy(dict(row,seed='0'),row)
    # A controller rejection can retain a fixed model that the gate did not accept.
    verify_successful_policy(dict(row,status='BUDGET_EXHAUSTED'),dict(row))
