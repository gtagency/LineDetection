function [next_p, next_theta ] = aoi_func(img, p0, theta0, box_width, box_height)
% aoi_func takes in a grayscale image(uint8) and a start point/angle
% generates next aoi(area of interest) and returns new point/angle


A = [1 0
     0 -1]; 
R = [cos(pi/2-theta0) -sin(pi/2-theta0)
     sin(pi/2-theta0) cos(pi/2-theta0)];

rot_img = rotateAround(img, p0(2), p0(1), rad2deg(pi/2-theta0));

left_x = round(p0(1) - box_width/2)
right_x = round(p0(1) + box_width/2)
bot_y = round(p0(2))
top_y = round(p0(2) - box_height)
size(img)

aoi = edge(rot_img(top_y:bot_y, left_x:right_x), 'canny');
num_points = sum(aoi(:));

[y x] = find(aoi==1);
aoi_points = [x'; y'];
aoi_points = aoi_points - repmat([box_width/2; box_height],1,size(aoi_points,2));
fit = polyfit(aoi_points(2,:),aoi_points(1,:),1);
x1 = polyval(fit, 0);
x2 = polyval(fit, -box_height);
p1 = [x2; -box_height];
p2 = [x1; 0];

p1 = A*R'*A*p1 + p0;
p2 = A*R'*A*p2 + p0;
next_p = p1;

if norm(p0-p1) < norm(p0-p2)
    next_p = p2;
end
next_theta = atan(fit(1)) + theta0;

end

