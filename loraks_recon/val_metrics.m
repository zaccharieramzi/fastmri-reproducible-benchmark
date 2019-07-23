val_path = '/media/Zaccharie/UHRes/singlecoil_val/';
filelist = dir(strcat(val_path, '*.h5'));
[n_files, un] = size(filelist);
AF = 4;
ssims = zeros(n_files);
psnrs = zeros(n_files);
for i = 1:n_files
    waitbar(i/ n_files)
    filename = filelist(i).name;
    kspaces = h5read(strcat(val_path, filename), '/kspace');
    p, s = reco(kspaces, AF);
    psnrs(i) = p;
    ssims(i) = s;
end
disp('Mean of PSNR', mean(psnrs))
disp('Std dev of PSNR', stddev(psnrs))
disp('Mean of SSIM', mean(ssims))
disp('Std dev of SSIM', stddev(ssims))