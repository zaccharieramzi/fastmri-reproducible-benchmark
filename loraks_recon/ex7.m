%% example 7: Single channel reconstruction exploration
clear;
close all;
clc
addpath('../.');  % Add the path containing the LORAKS functions
warning('off','MATLAB:pcg:tooSmallTolerance');

%% single-channel T2-weighted data
% load T2_single_channel % Load k-space data
% 
val_path = '/media/Zaccharie/UHRes/singlecoil_val/';
filelist = dir(strcat(val_path, '*.h5'));

selected_file = 1;
filename = filelist(selected_file).name;
kspaces = h5read(strcat(val_path, filename), '/kspace');
kspaces = complex(kspaces.r, kspaces.i);
kspaces = permute(kspaces, [2 1 3]);
selected_slice = 15;
kspace = kspaces(:, :, selected_slice);
kData = kspace .*  sqrt(numel(kspace));
% remove zero-padding and oversampled A/D
kData = fftshift(ifft(ifftshift(kData,1),[],1),1);
kData = fftshift(fft(ifftshift(kData(end/2+1+[-end/4:end/4-1],:,:))));
kData = kData(:,19:350);

% baboon exp
% path = 'baboon.h5';
% kspace = h5read(path, '/baboon_kspace');
% kspace = complex(kspace.r, kspace.i);
% kData = kspace;


% Display gold standard
figure;
imshow(abs(fftshift(ifft2(ifftshift(kData)))), [])
title('Gold Standard');

image = abs(fftshift(ifft2(ifftshift(kData))));


str = 'random with ACS';
% kMask = kMaskRandACS2;
mask = gen_mask(kData,4);
kspace_size = size(kData);
mask = repmat(mask, [1 kspace_size(1)]);
kMask = permute(mask, [2 1]);
rankACLORAKS = 15;


disp('********************************************************************');
disp(['Sampling: ' str]);
disp('********************************************************************');

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
ssim_recon  = ssim(im_recon, image, 'DynamicRange',max(image, [], 'all') - min(image, [], 'all'));
ssim_zero_filled = ssim(zero_filled, image, 'DynamicRange', max(image, [], 'all') - min(image, [], 'all'));

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