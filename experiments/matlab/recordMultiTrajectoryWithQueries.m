clear;
close all;
addpath('drawing_related/');
f1 = figure;
figure(f1);

global numObj trajectories mouseDown trajNum queries genEnv;
numObj = 3;
% genEnv = @plotTargetRandom;
% genEnv = @plotRandViaPointTask;
genEnv = @plotRandHeightObj;

%q = plotTargetRandom(numObj);
%q = plotRandViaPointTask();
q = genEnv();
hold on

trajectories = {};
mouseDown = false;
trajNum = 0;
queries = [q];

set(gcf, 'WindowButtonMotionFcn', @mouseMove_multi);
set(gcf,'WindowButtonDownFcn', @record_multi, 'WindowButtonUpFcn',   @stop_multi_record);


