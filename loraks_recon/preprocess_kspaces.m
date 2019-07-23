function [kspaces_out] = preprocess_kspaces(kspaces_in)

kspaces = complex(kspaces_in.r, kspaces_in.i);
kspaces = permute(kspaces, [2 1 3]);
kData = kspaces .*  sqrt(numel(kspaces(1, :, :)));
% remove zero-padding and oversampled A/D
kData = fftshift(ifft(ifftshift(kData,1),[],1),1);
kData = fftshift(fft(ifftshift(kData(end/2+1+[-end/4:end/4-1],:,:))));
% identify and remove zero-padding (works before applying the mask)
[ii, jj, kk] = find(~ real(kData));
[count, values] = hist(kk, unique(kk));
k_size = size(kData);
zero_values = values(count(:) == k_size(2));
kData = kData(:,:,~zero_values);
end

