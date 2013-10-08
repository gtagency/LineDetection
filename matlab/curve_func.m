function [points thetas] = curve_func(img, p0, theta0, aoi_width, aoi_height, num)
% curve_func takes an image and initial point/angle
% attempts to follow curve using num boxes of size (aoi_width, aoi_height).

img = rgb2gray(img);
points = zeros(2,4);
thetas = zeros(1,4);
p = p0;
theta = theta0;

for i=1:num
    points(:,i) = p;
    thetas(i) = theta;
    [p theta] = aoi_func(img, p, theta, aoi_width, aoi_height);
end
points(:,i+1) = p;
thetas(i+1) = theta;
