function [img_cropped] = crop_center(image,crop)
%CROP_CENTER Summary of this function goes here
%   Detailed explanation goes here
[x y] = size(image);
startx = int16(x/2) - int16(crop/2);
starty = int16(y/2) - int16(crop/2);
img_cropped = image(startx + 1:startx+crop, starty + 1:starty+crop);
end

