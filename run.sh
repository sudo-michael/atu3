#!bin/bash

# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 1 --buffer-size 100_000 --penalize-jerk --group pj_air3d
# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 2 --buffer-size 100_000 --penalize-jerk --group pj_air3d
# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 3 --buffer-size 100_000 --penalize-jerk --group pj_air3d


python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 1 --buffer-size 100_000  --group no_pj_air3d
python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 2 --buffer-size 100_000  --group no_pj_air3d
python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 3 --buffer-size 100_000  --group no_pj_air3d

