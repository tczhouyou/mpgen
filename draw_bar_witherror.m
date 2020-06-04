clear
num_type = 4;
num_comp = 3;
filenames = {'original_mdgan', 'entropy_mdgan', 'original_mdn', 'entropy_mdn'};
means = zeros(num_type,num_comp);
stds = zeros(num_type,num_comp);

for i = 1 : length(filenames)
    filename = filenames{i};
    data = importdata(filename);
    data = sort(data);
%     data = data(2:end-1,:);
    means(i,:) = mean(data);
    stds(i,:) = std(data);
end

figure
hold on
hb = bar(1:num_comp,means');
pause(0.1); %pause allows the figure to be created
for ib = 1:numel(hb)
    xData = hb(ib).XData+hb(ib).XOffset;
    errorbar(xData,means(ib,:),stds(ib,:),'k.')
end