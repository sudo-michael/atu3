#!bin/bash
python atu3/new_sac.py --use-hj --track --capture-video
python atu3/new_sac.py --use-hj --track --capture-video --normalize-obs
python atu3/new_sac.py --use-hj --track --capture-video --total-timesteps 2_000_000
python atu3/new_sac.py --use-hj --track --capture-video --total-timesteps 2_000_000 --normalize-obs
