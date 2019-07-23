function [kspaces_out] = preprocess_kspaces(kspaces_in)

kspaces = complex(kspaces_in.r, kspaces_in.i);
kspaces = permute(kspaces, [3 2 1]);
kData = kspaces .*  sqrt(numel(kspaces(1, :, :)));
k_size = size(kData);
k_down_sampled = zeros(k_size(1), k_size(2) / 2, k_size(3));
% remove zero-padding and oversampled A/D
for i = 1:k_size(1)
    k_slice = shiftdim(kData(i, :, :), 1);
    k_slice = fftshift(ifft(ifftshift(k_slice,1),[],1),1);
    k_slice = fftshift(fft(ifftshift(k_slice(end/2+1+[-end/4:end/4-1],:,:))));
    k_down_sampled(i, :, :) = k_slice;
end
% identify and remove zero-padding (works before applying the mask)
[ii, jj] = find(~ real(shiftdim(k_down_sampled(1, :, :), 1)));
[count, values] = hist(jj, unique(jj));
k_size = size(k_down_sampled);
zero_values = values(count(:) == k_size(2));
indexes = ones([k_size(3) 1]);
indexes(zero_values) = 0;
non_zero_values = find(indexes);
kspaces_out = k_down_sampled(:,:,non_zero_values);
end

