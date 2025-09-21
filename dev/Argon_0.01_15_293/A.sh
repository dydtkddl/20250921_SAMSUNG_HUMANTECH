python ../MAIN_AL_QT_XYscale_control_NNmodified_nplog_rdopt.py  --data ../Ar_293K/Ar_293K_Ar_293_0.01_to_Ar_293_15_dataset.csv  --hidden_dim1 128 --hidden_dim2 64 --rd_frac 1 --num_threads 30 --x_scale  --qt_frac 0 --activation gelu --target_ratio 0.8  --samples_per_iter 30 --dropout_rate 0.05
python ../MAIN_AL_QT_XYscale_control_NNmodified_nplog_rdopt.py  --data ../Ar_293K/Ar_293K_Ar_293_0.01_to_Ar_293_15_dataset.csv  --hidden_dim1 128 --hidden_dim2 64 --rd_frac 0 --num_threads 30 --x_scale  --qt_frac 0.2 --activation gelu --target_ratio 0.8  --samples_per_iter 30 --dropout_rate 0.05

