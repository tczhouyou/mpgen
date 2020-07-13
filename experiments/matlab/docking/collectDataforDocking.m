clear;
close all;
addpath('recording_events/');
addpath('../drawObj/');
f1 = figure;
figure(f1);

global trajectories mouseDown trajNum cquery queries

queries = [];
trajectories = {};
mouseDown = false;
trajNum = 0;
cquery = plotDockingStations();

hold on
set(gcf, 'WindowButtonMotionFcn', @mouseMoveEvent);
set(gcf,'WindowButtonDownFcn', @recordEvent, 'WindowButtonUpFcn',   @stopEvent);


