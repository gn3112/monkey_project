function [single_line] = BMITestNN3(parameters, observation)
weights = parameters.weights;
%num_of_loops = parameters.num_of_loops;
%predict position for one specific trial and angle
%observation = neural_data_cell{any};
% hold on 
% [single_line] = BMITestNN2(parameters, neural_data_cell{1});
% plot3(single_line(:,1),single_line(:,2),single_line(:,3));
% [single_line] = BMITestNN2(parameters, neural_data_cell{2});
% plot3(single_line(:,1),single_line(:,2),single_line(:,3));
% [single_line] = BMITestNN2(parameters, neural_data_cell{3});
% plot3(single_line(:,1),single_line(:,2),single_line(:,3));
% [single_line] = BMITestNN2(parameters, neural_data_cell{4});
% plot3(single_line(:,1),single_line(:,2),single_line(:,3));
% [single_line] = BMITestNN2(parameters, neural_data_cell{5});
% plot3(single_line(:,1),single_line(:,2),single_line(:,3));

sizee = size(observation,1);
observation = [observation, eye(sizee,1)];

number_of_nodes = [100, 100, 2];
number_of_labels = 2;
number_of_layers = length(number_of_nodes);

unit_outputs = cell(1,number_of_layers);

for laynum = 1:number_of_layers
    if laynum ~= number_of_layers
        unit_outputs{laynum} = zeros(number_of_nodes(laynum),1);
        %to multiply with bias 
        unit_outputs{laynum}(end) = 1;
    else
        unit_outputs{laynum} = zeros(number_of_labels,1);
    end
end

binsize = 20;
%predict 3D position
%number of loops should depend on the observation
trial_time = size(observation,1);
adjusted_trial_time = floor(trial_time/binsize)*binsize;
num_of_loops = (adjusted_trial_time - 300)/binsize;
predictions = zeros(num_of_loops,2);
for loop_number = 0:num_of_loops-1
    for layer_num = 1:number_of_layers  
        % gotta loop over layers, for each iteration of a loop over examples
        if layer_num == 1
            %first layer
            layer_output = sum(observation(loop_number*binsize+1:loop_number*binsize + 300,:))*weights{layer_num};
            layer_output = layer_output/300;
            unit_outputs{layer_num}(1:end-1) = layer_output';
            unit_outputs{layer_num}(1:end-1) = sigmoid(unit_outputs{layer_num}(1:end-1));
        elseif layer_num == number_of_layers
            %compute final layer activation
            layer_output = unit_outputs{layer_num-1}'*weights{layer_num};
            unit_outputs{layer_num} = layer_output';
            %dont compute anything using softmax, cuz use regression
            %compute the last layer output using softmax
%             unit_outputs(layer_num).a = softmax(unit_outputs(layer_num).a);
        else
            %compute hidden layer activation
            layer_output = unit_outputs{layer_num-1}'*weights{layer_num};
            unit_outputs{layer_num}(1:end-1) = layer_output';
            unit_outputs{layer_num}(1:end-1) = sigmoid(unit_outputs{layer_num}(1:end-1));
        end
    end

predictions(loop_number+1,:) = unit_outputs{number_of_layers}';
end

%have to add the predictions to the starting point and then interpolate
points = zeros(size(predictions,1)+1,2);
for i = 1:size(points,1)-1
    points(i+1,:) = points(i,:) + predictions(i,:);
end
%will have 1 less interpolations than points
n = 100;
t = linspace(0,1,n)';
interpolations = cell(1,size(points,1)-1);
for i = 1:numel(interpolations)
    interpolations{i} = (1-t)*points(i,:) + t*points(i+1,:);
end
single_line = [];
for i = 1:numel(interpolations)
    single_line = [single_line; interpolations{i}];
end

%compute RMSE

function hidden_unit_out = sigmoid(x)
    alpha = 2.25;
    % can be applied to scalars and vectors
    hidden_unit_out = 1./(1+exp(-alpha*x));
end

% function last_layer_out = softmax(input)
%     last_layer_out = exp(input)/(sum(exp(input)));
% end

%[value, prediction] = TestNN(parameters, partitioned_data.test_inp(1,:))
%partitioned_data.test_label(1,:)
%prediction = TestNN(parameters, partitioned_data.test_inp);
%prediction = TestNN(parameters, test_score(1:30));
end