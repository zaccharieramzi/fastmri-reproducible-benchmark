function [p,s] = reco(kspaces, AF)
kspaces = preprocess_kspaces(kspaces);
kspace_size = size(kspaces);
images = zeros(kspace_size);
rec_images = zeros(kspace_size);
mask = gen_mask(shiftdim(kspaces(1, :, :), 1), AF);
mask = repmat(mask, [1 kspace_size(2)]);
kMask = permute(mask, [2 1]);
for i = 1:kspace_size(1)
   kspace = shiftdim(kspaces(i, :, :), 1);
   images(i, :, :) = abs(fftshift(ifft2(ifftshift(kspace))));
   undersampledData = kspace.*kMask;
   recon = AC_LORAKS(undersampledData, kMask, 15, 5, 'S', [], [], [], 15);
   rec_images(i, :, :) = abs(fftshift(ifft2(ifftshift(recon))));
end
p = psnr(rec_images, images, max(images, [], 'all'));
s = ssim(rec_images, images, 'DynamicRange',max(images, [], 'all') - min(images, [], 'all'));
end

