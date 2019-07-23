function [mask] = gen_mask(kspace,accel_factor)
%GEN_MASK Summary of this function goes here
%   Detailed explanation goes here
kspace_size = size(kspace);
n_samples = int16(kspace_size(2) / accel_factor);
mask = zeros([kspace_size(2) 1]);
n_center = int16(kspace_size(2) * (32 / accel_factor) / 100);
n_non_center = n_samples - n_center;
left_center = int16(kspace_size(2)/2 - n_center/2);
right_center = int16(kspace_size(2)/2 + n_center/2);
mask(left_center:right_center) = 1;
non_center_indices = 1:kspace_size(2);
non_center_indices = non_center_indices(non_center_indices < left_center | non_center_indices > right_center);
selected_indices = randsample(non_center_indices, n_non_center);
mask(selected_indices) = 1;
end

