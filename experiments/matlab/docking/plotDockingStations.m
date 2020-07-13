function q = plotRandOriGoalTask()
axis([-5 25 -5 25])
hold on
pbaspect([1 1 1])
qs = pi/2;%rand(1) * 2 * pi;
qg = pi + rand(1) * pi/2;


start = [0,0];% + rand(1,2) * 9;
goal = [10,10] - rand(1,2) * 5;

plot(start(1), start(2), 'r.', 'MarkerSize',20);
plot(goal(1), goal(2), 'b.', 'MarkerSize',20);

width = 5/3;
length = 3;


drawOpenRec(goal, qg, width, length, 'b-')
drawOpenRec(start, qs, width, length, 'r-')

q = [start, qs, goal, qg];

end