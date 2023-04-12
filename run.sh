#!bin/bash

# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 1 --buffer-size 100_000 --penalize-jerk --group pj_air3d
# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 2 --buffer-size 100_000 --penalize-jerk --group pj_air3d
# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 3 --buffer-size 100_000 --penalize-jerk --group pj_air3d


# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 1 --buffer-size 100_000  --group no_pj_air3d
# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 2 --buffer-size 100_000  --group no_pj_air3d
# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 3 --buffer-size 100_000  --group no_pj_air3d

# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 750_000 --gamma 0.9 --capture-video --seed 1 --buffer-size 100_000 
# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 750_000 --gamma 0.9 --capture-video --seed 2 --buffer-size 100_000 
# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 750_000 --gamma 0.9 --capture-video --seed 3 --buffer-size 100_000 

#python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 750_000 --gamma 0.9 --capture-video --seed 1 --buffer-size 125_000 
#python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 750_000 --gamma 0.9 --capture-video --seed 2 --buffer-size 125_000 
#python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 750_000 --gamma 0.9 --capture-video --seed 3 --buffer-size 125_000 
#
#python atu3/sac.py --reward-shape-hj-takeover 0.005 --track --total-timesteps 750_000 --gamma 0.9 --capture-video --seed 1 --buffer-size 125_000 
#python atu3/sac.py --reward-shape-hj-takeover 0.005 --track --total-timesteps 750_000 --gamma 0.9 --capture-video --seed 2 --buffer-size 125_000 
#python atu3/sac.py --reward-shape-hj-takeover 0.005 --track --total-timesteps 750_000 --gamma 0.9 --capture-video --seed 3 --buffer-size 125_000 
#
#python atu3/sac.py --reward-shape-hj-takeover 0.005 --track --total-timesteps 750_000 --gamma 0.95 --capture-video --seed 1 --buffer-size 125_000 
#python atu3/sac.py --reward-shape-hj-takeover 0.005 --track --total-timesteps 750_000 --gamma 0.95 --capture-video --seed 2 --buffer-size 125_000 
#python atu3/sac.py --reward-shape-hj-takeover 0.005 --track --total-timesteps 750_000 --gamma 0.95 --capture-video --seed 3 --buffer-size 125_000 


# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 1 --buffer-size 125_000 --group no_pj_air3d_3
# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 2 --buffer-size 125_000 --group no_pj_air3d_3
# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 3 --buffer-size 125_000 --group no_pj_air3d_3

# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 1 --buffer-size 125_000 --group pj_air3d_3 --penalize-jerk
# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 2 --buffer-size 125_000 --group pj_air3d_3 --penalize-jerk
# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 3 --buffer-size 125_000 --group pj_air3d_3 --penalize-jerk
# python atu3/sac.py --reward-shape-hj-takeover 0.05 --track --total-timesteps 1_000_000 --gamma 0.9 --capture-video --seed 1 --buffer-size 125_000
# python atu3/td3bc.py --track --capture-video --seed 1
# python atu3/td3bc.py --track --capture-video --seed 2
# python atu3/td3bc.py --track --capture-video --seed 3
# python atu3/td3bc.py --track --capture-video --use-bc --seed 1
# python atu3/td3bc.py --track --capture-video --use-bc --seed 2
# python atu3/td3bc.py --track --capture-video --use-bc --seed 3


python atu3/new_sac.py --use-hj --track --capture-video
python atu3/new_sac.py --use-hj --track --capture-video --normalize-obs
python atu3/new_sac.py --use-hj --track --capture-video --total-timesteps 2_000_000
python atu3/new_sac.py --use-hj --track --capture-video --total-timesteps 2_000_000 --normalize-obs
