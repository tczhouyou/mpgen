function [trajs, idx] = processAll(trajectories, basicFile)

trajs = cell(length(trajectories),1);

dim = size(trajectories{1}, 2);

k = 1;
idx = [];
for i = 1 : length(trajectories)
    ctraj = trajectories{i};
%     if size(ctraj,1) < 30
%         continue;
%     end
    traj = processTrajectory(trajectories{i});
    
    for j = 2 : size(traj,2)
        traj(:,j) = smoothdata(traj(:,j), 'gaussian');
    end
    
    if nargin == 2
        filename = strcat(basicFile, '_', num2str(k), '.csv');
        k = k + 1;
        idx = [idx,i];
        dlmwrite(filename, traj)
    end
    trajs{i} = traj;
    
    for j = 1 : dim
        subplot(dim,1,j)
        hold on
        plot(traj(:,1), traj(:,j+1))
        title(strcat('dim ', num2str(j)))
    end
end





end

