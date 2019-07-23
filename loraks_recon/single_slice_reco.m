
%%
val_path = '/media/Zaccharie/UHRes/singlecoil_val/';
filelist = dir(strcat(val_path, '*.h5'));

selected_file = 1;
filename = filelist(selected_file).name;
kspaces = h5read(strcat(val_path, filename), '/kspace');
kspaces = complex(kspaces.r, kspaces.i);
kspaces = permute(kspaces, [2 1 3]);
images = h5read(strcat(val_path, filename), '/reconstruction_esc');
images = permute(images, [2 1 3]);
disp('Data loaded')
%%

selected_slice = 10;
image = images(:, :, selected_slice);
kspace = kspaces(:, :, selected_slice);
kspace = kspace .*  sqrt(numel(kspace));
kspace_size = size(kspace);
accel_factor = 4;
mask = gen_mask(kspace,accel_factor);
mask = repmat(mask, [1 kspace_size(1)]);
mask = permute(mask, [2 1]);
masked_kspace = mask .* kspace;
disp('Mask generated')

%%
% ranks = [1 2 4 10 15 20 30 40 50];
ranks = [1 2 4 10 15 20];
psnrs = zeros([size(ranks) 1]);
kspace_mses = zeros([size(ranks) 1]);
for i = 1:length(ranks)
    rank = ranks(i)
    max_iter = 15;
    k_recon = AC_LORAKS(masked_kspace, mask, rank, 3, 'S', [], [], [], max_iter);
    im_recon = fftshift(ifft2(ifftshift(k_recon)));
    cropped_im_recon = crop_center(abs(im_recon), 320);
    psnrs(i) = psnr(cropped_im_recon, image, max(image, [], 'all'));
    kspace_mses(i) = mean(abs((kspace - k_recon).^2), 'all');
end
figure(3)
plot(ranks, psnrs)
hold on
yyaxis right
plot(ranks, kspace_mses)
hold off

%%

lambda = 1e-5;
max_iter = 15;
k_recon = AC_LORAKS(masked_kspace, mask, 20, 3, 'S', [], [], max_iter);
im_recon = fftshift(ifft2(ifftshift(k_recon)));

cropped_im_recon = crop_center(abs(im_recon), 320);

%%
disp('Psnr')
[peaksnr, snr] = psnr(cropped_im_recon, image, max(image, [], 'all'));
disp(peaksnr)
disp('SSIM')
[ssimval, ssimmap] = ssim(cropped_im_recon, image, 'DynamicRange', max(image, [], 'all') - min(image, [], 'all'));
disp(ssimval)

figure(1)
imshow(image, [])
figure(2)
imshow(cropped_im_recon, [])

%%
zero_filled = fftshift(ifft2(ifftshift(masked_kspace)));
cropped_zero_filled = crop_center(abs(zero_filled), 320);
disp('Psnr zero filled')
[peaksnr_z, snr_z] = psnr(cropped_zero_filled, image, max(image, [], 'all'));
disp(peaksnr_z)
disp('SSIM zero filled')
[ssimval, ssimmap] = ssim(cropped_zero_filled, image, 'DynamicRange', max(image, [], 'all') - min(image, [], 'all'));
disp(ssimval)




