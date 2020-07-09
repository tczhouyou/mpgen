function q = plotRandHeightObj()
axis([-10 10 -0.1 10])
hold on

height = rand * 8;


plot(linspace(-10,10,100), zeros(1,100), 'k-', 'LineWidth', 3);

plot(-5, 0, 'ro', 'MarkerSize', 10);
plot(5, 0, 'bo', 'MarkerSize', 10);

points = [-1 * ones(100,1), linspace(0,height,100)'];
points = [points; linspace(-1,1,10)', height * ones(10,1)];
points = [points; ones(100,1), linspace(height,0,100)'];
points = [points; linspace(1,-1,10)', zeros(10,1)];

patch(points(:,1), points(:,2), [0.5,0.5,0.5]);

q = height;




end