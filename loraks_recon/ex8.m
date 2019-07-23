%% example 7: Single channel reconstruction exploration
clear;
close all;
clc
addpath('../.');  % Add the path containing the LORAKS functions
warning('off','MATLAB:pcg:tooSmallTolerance');

%% single-channel T2-weighted data
% path to the h5 file
file_path = 'file1000000.h5';
kspaces = h5read(file_path, '/kspace');
kspaces = complex(kspaces.r, kspaces.i);
kspaces = permute(kspaces, [2 1 3]);
% slice selection
selected_slice = 15;
kspace = kspaces(:, :, selected_slice);
kData = kspace .*  sqrt(numel(kspace));
image = abs(fftshift(ifft2(ifftshift(kData))));


% Display gold standard
figure;
imshow(image, [])
title('Gold Standard');



str = 'random with ACS';
mask = gen_mask(kData,2);
kspace_size = size(kData);
mask = repmat(mask, [1 kspace_size(1)]);
kMask = permute(mask, [2 1]);
rankACLORAKS = 15;


% Display sampling pattern
figure;
imagesc(kMask);
axis equal;axis off;colormap(gray);
undersampledData = kData.*kMask;
title(['Sampling: ' str]);


% AC-LORAKS reconstruction with Eq. (13).  (S-matrix, exact data consistency)
tic
recon = AC_LORAKS(undersampledData, kMask, rankACLORAKS, 5, 'S', [], [], [], 15);
time = toc;

im_recon = abs(fftshift(ifft2(ifftshift(recon))));
zero_filled = abs(fftshift(ifft2(ifftshift(undersampledData))));
psnr_recon  = psnr(im_recon, image, max(image, [], 'all'));
psnr_zero_filled = psnr(zero_filled, image, max(image, [], 'all'));

% Display results
figure;
imshow(im_recon, []);
title(['AC-LORAKS, ' str ', PSNR = ' num2str(psnr_recon) ', time = ' num2str(time) ' seconds']);
disp(' ');

% Display results
figure;
imshow(zero_filled, []);
title(['Zero-filled PSNR = ' num2str(psnr_zero_filled)]);
disp(' ');