clear;
close all;
addpath('drawing_related/');
f1 = figure;
figure(f1);

global numObj
numObj = 3;

%q = plotTargetRandom(numObj);
q = plotRandViaPointTask();
hold on

global trajectories;
global mouseDown;
global trajNum;
global queries;

trajectories = {};
mouseDown = false;
trajNum = 0;
queries = [q];

set(gcf, 'WindowButtonMotionFcn', @mouseMove_multi);
set(gcf,'WindowButtonDownFcn', @record_multi, 'WindowButtonUpFcn',   @stop_multi_record);


