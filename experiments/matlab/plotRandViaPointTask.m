function q = plotRandViaPointTask()
axis([-10 10 -10 10])
hold on

q = [];

sx_max = 3;
sx_min = -3;
sy_max = 9;
sy_min = -9;

x = (sx_max - sx_min) * rand(1) + sx_min;
y = (sy_max - sy_min) * rand(1) + sy_min;
th = pi * rand(1) - pi/2;

quiver(x, y, 1*cos(th),1*sin(th), 'o');
q = [q, x, y, th];

sx_max = -4;
sx_min = -9;
sy_max = 9;
sy_min = -9;

x = (sx_max - sx_min) * rand(1) + sx_min;
y = (sy_max - sy_min) * rand(1) + sy_min;
quiver(x, y, 1,0, 'o');
q = [q, x, y];

sx_max = 9;
sx_min = 4;
sy_max = 9;
sy_min = -9;

x = (sx_max - sx_min) * rand(1) + sx_min;
y = (sy_max - sy_min) * rand(1) + sy_min;
quiver(x, y, 1,0, 'o');
q = [q, x, y];

end