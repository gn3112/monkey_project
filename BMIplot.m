%plot3(single_line(:,1),single_line(:,2),single_line(:,3));
%plot3(positions_cell{1}(:,1),positions_cell{1}(:,2),positions_cell{1}(:,3));
%plot(single_line(:,1),single_line(:,2))

% predictions is num_of_bins(49) - 15 by 3 vector
% have 980 overall timesteps, want our single line to have as many points
% have 33 points in predictions for interpolation
% hold on 
% for i = 1:9
% [single_line] = BMITestNN2(parameters, neural_data_cell{i});
% plot3(single_line(:,1),single_line(:,2),single_line(:,3));
% end
figure(1)
hold on 
for i = 1:8
[single_line] = BMITestNN3(parameters, neural_data_cell{i});
plot(single_line(:,1),single_line(:,2), 'linewidth', 1);
end
title('Predicted','fontsize',15)
legend('1','2','3','4','5','6','7','8')

xlabel('X','fontsize',15) 
ylabel('Y','fontsize',15)
zlabel('Z','fontsize',15)

figure(2)
hold on 
for i = 1:8
plot(positions_cell{i}(:,1),positions_cell{i}(:,2))
end
title('Actual','fontsize',15)
%for the actual trajectories
%{
hold on 

for i = 1:9
plot3(positions_cell{i}(:,1),positions_cell{i}(:,2),positions_cell{i}(:,3));
end

xlabel('X','fontsize',15) 
ylabel('Y','fontsize',15)
zlabel('Z','fontsize',15)
%}