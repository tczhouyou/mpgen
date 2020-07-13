close all
numRow = 5;
numData = numRow * numRow;
idx = randsample(length(trajectories), numData);
tqs = queries(idx,:);
ttrj = trajectories(idx);
th = 0 : pi/60 : 2 *pi;

numObs = (size(tqs,2) - 4) / 2;
for i = 1 : numData
    subplot(numRow, numRow, i);
    axis([-10,10,-10,10]);
    hold on;
     cq = tqs(i,:);
%     for j = 1 : numObs
%         xunit = 3 * cos(th) + cq((j-1)*2+1);
%         yunit = 3 * sin(th) + cq((j-1)*2+2);
%         plot(xunit, yunit, 'r-'); 
%     end
%     plot(cq(end-1), cq(end), 'bo');
%     plot(cq(end-3), cq(end-2), 'ro');
%     plot(ctraj(:,1), ctraj(:,2), 'k-');

     height = cq(1);
    points = [-1 * ones(100,1), linspace(0,height,100)'];
    points = [points; linspace(-1,1,10)', height * ones(10,1)];
    points = [points; ones(100,1), linspace(height,0,100)'];
    points = [points; linspace(1,-1,10)', zeros(10,1)];
    patch(points(:,1), points(:,2), [0.5,0.5,0.5]);   
    
    ctraj = ttrj{i};
    plot(ctraj(:,1), ctraj(:,2), 'k-');
end