close all
numRow =2;
numData = numRow * numRow;
idx = randsample(length(trajs), numData);
tqs = queries(idx,:);
ttrj = trajectories(idx);
th = 0 : pi/60 : 2 *pi;

numObs = (size(tqs,2) - 4) / 2;
for i = 1 : 2
    subplot(numRow, numRow, i);
    axis([-5,25,-5,25]);
    hold on;
    cq = tqs(i,:);
        
    drawOpenRec(cq(4:5), cq(6), 5/3, 3, 'b-')
    drawOpenRec(cq(1:2), cq(3), 5/3, 3, 'r-')
    
    ctraj = ttrj{i};
    plot(ctraj(:,1), ctraj(:,2), 'k-');
    
end