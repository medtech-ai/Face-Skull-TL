python3 train_ddgan.py --dataset masks --exp test_run --num_channels 1 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 2 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --num_proc_node 2 --num_process_per_node 1 \
--ch_mult 1 2 2 2 --save_content --image_size 32