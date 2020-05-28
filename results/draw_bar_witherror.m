
bl = importdata('result_uniform.csv');
bl = sort(bl);
bl = bl(2:end-1,:);
gpr = importdata('result_gpr.csv');
% gpr = sort(gpr);
% gpr = gpr(2:end-1,:);
omdn = importdata('result_orig.csv');
omdn = sort(omdn);
omdn = omdn(2:end-1,2:end);
emdn = importdata('result_entropy.csv');
emdn = sort(emdn);
emdn = emdn(2:end-1,2:end);

means = [mean(bl); mean(gpr); mean(omdn); mean(emdn)]; % mean velocity
stds = [std(bl); std(gpr); std(omdn); std(emdn)];
figure
hold on
hb = bar(1:4,means');
pause(0.1); %pause allows the figure to be created
for ib = 1:numel(hb)
    xData = hb(ib).XData+hb(ib).XOffset;
    errorbar(xData,means(ib,:),stds(ib,:),'k.')
end