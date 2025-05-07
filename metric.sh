root="/home/user/dyh/ts/Diffusion-TS/OUTPUT/"
name="ETTh_x_x_mark_best"
cfg="etth"

python /home/user/dyh/ts/Diffusion-TS/Experiments/metric_pytorch.py --root "$root$cfg" \
--ori_path "$root/samples/${cfg}_norm_truth_24_train.npy" \
--fake_path "$root$name/ddpm_fake_${name}.npy"