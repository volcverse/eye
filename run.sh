OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 \
   train_eyeReal.py --exp_name lego_bulldozer10000_scale_0.083_R_10_150_FOV_40_theta_40_140_phi_60_120_aux2_10_mutex05 \
    --data_path dataset/scene_data/lego_bulldozer/lego_bulldozer10000_scale_0.083_R_10_150_FOV_40_theta_40_140_phi_60_120 \
    --image_height 1080 --image_width 1920 --embed_dim 16 --random_ratio 1 \
    --workers 2 --T_scale 1.0 --weight-decay 5e-5 --lr 0.0004 -b 2 --epoch 20 --kernel_size 3 \
    --FOV 40 \
    --l1_mutex_ratio 0.5 \
    --aux_loss --aux_ratio 0.3 --aux_weight 10 \
    --l1_mutex --scene lego_bulldozer