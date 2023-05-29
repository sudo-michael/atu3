#!/bin/bash
python run_experiment.py \
    --mode='train' \
    --experiment_class='VerifyDeepReach' \
    --experiment_name='atu_multivehiclecollisionavoidtube_1' \
    --minWith='target' \
    --dynamics_class='MultiVehicleCollision' \
    --adj_rel_grads=True --diff_model=False \
    --seed=0 \
    --batch_size=1 \
    --clip_grad=0.0 \
    --counter_end=100000 \
    --counter_start=0 \
    --epochs_til_ckpt=1000 \
    --experiment_dir='./runs' \
    --lr=2e-05 \
    --model='sine' \
    --model_mode='mlp' \
    --num_epochs=160000 \
    --num_hl=3 \
    --num_nl=512 \
    --num_src_samples=10000 \
    --numpoints=65000 \
    --pretrain \
    --pretrain_iters=60000 \
    --seed=0 \
    --steps_til_summary=100 \
    --tMax=1.0 \
    --tMin=0.0 \
    --val_time_resolution=3 \
    --val_x_resolution=200 \
    --val_y_resolution=200 \
    --val_z_resolution=5 \