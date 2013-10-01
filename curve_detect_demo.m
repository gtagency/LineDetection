clear
clc

box_height = 40;
box_width = 20;
start_p = [350; 705];
start_theta = pi/2.7;
num = 10;

rgb_frame = imread('images/sample_image.png');
imshow(rgb_frame);
hold on

[points thetas] = curve_func(rgb_frame, start_p, start_theta, box_width, box_height, num);
plot(points(1,:),points(2,:),'b','LineWidth',1.5);
