f = waitbar(0,'Please wait...');
val_path = '/media/Zaccharie/UHRes/singlecoil_val/';
filelist = dir(strcat(val_path, '*.h5'));
[n_files, un] = size(filelist);
AF = 8;
ssims = zeros([n_files, 1]);
psnrs = zeros([n_files, 1]);
% for i = 1:n_files
parfor i = 1:n_files
    filename = filelist(i).name;
    kspaces = h5read(strcat(val_path, filename), '/kspace');
    [p, s] = reco(kspaces, AF);
    psnrs(i) = p;
    ssims(i) = s;
    waitbar(i/ n_files, f, i)
end
disp(strcat('Mean of PSNR :', num2str(mean(psnrs))))
disp(strcat('Std dev of PSNR :', num2str(std(psnrs))))
disp(strcat('Mean of SSIM :', num2str(mean(ssims))))
disp(strcat('Std dev of SSIM :', num2str(std(ssims))))
close(f)