clear;
close all;
f1 = figure;
figure(f1);


% plotTargets(3);
axis([-2,2,-2,2]);
hold on

global trajectories;
global mouseDown;
global trajNum;

trajectories = {};
mouseDown = false;
trajNum = 0;

set(gcf, 'WindowButtonMotionFcn', @mouseMove_multi);
set(gcf,'WindowButtonDownFcn', @record_multi, 'WindowButtonUpFcn',   @stop_multi);


