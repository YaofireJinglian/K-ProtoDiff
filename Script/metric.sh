
python metric_pytorch.py --root "$root$cfg" \
--ori_path "$root/samples/${cfg}_norm_truth_24_train.npy" \
--fake_path "$root$name/ddpm_fake_${name}.npy"