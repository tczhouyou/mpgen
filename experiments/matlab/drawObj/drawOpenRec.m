function drawOpenRec(pos, angle, width, length, linestyle)

plot(pos(1), pos(2), 'k.', 'MarkerSize', 20);
hold on
pbaspect([1 1 1])

r = angle;
wx = cos(pi/2 + r) * width;
wy = sin(pi/2 + r) * width;
lx = cos(r) * length;
ly = sin(r) * length;


p1 = [pos(1) + wx/2, pos(2) + wy/2];
p2 = [pos(1) - wx/2, pos(2) - wy/2];
p1 = [p1(1) - 2 * lx/5, p1(2) - 2 * ly/5];
p2 = [p2(1) - 2 * lx/5, p2(2) - 2 * ly/5];

p3 = [p2(1) + lx, p2(2) + ly];
p4 = [p3(1) + wx/5, p3(2) + wy/5];
p5 = [p4(1) - 4 * lx/5, p4(2) - 4 * ly/5];
p6 = [p5(1) + 3 * wx/5, p5(2) + 3 * wy/5];
p7 = [p6(1) + 4 * lx/5, p6(2) + 4 * ly/5];
p8 = [p7(1) + wx/5, p7(2) + wy/5];

points = [p1;p2;p3;p4;p5;p6;p7;p8;p1];
plot(points(:,1), points(:,2), linestyle, 'LineWidth', 2);

end

