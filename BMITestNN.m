function [single_line] = BMITestNN(parameters, observation)
weights = parameters.weights;
num_of_bins = parameters.num_of_bins;
%predict position for one specific trial and angle
%observation = neural_data_cell{any};
%[single_line] = BMITestNN(parameters, neural_data_cell{1});

% means = zeros(1,size(observation,2));
% stds = zeros(1,size(observation,2));
% for feature = 1:size(observation,2)
%     means(feature) = mean(observation(:,feature));
%     stds(feature) = std(observation(:,feature));
% end
% % Normalizing every features data:
% for i = 1:size(observation,2)
%     observation(:,i) = ((observation(:,i) - means(i))./ stds(i));
% end

sizee = size(observation,1);
observation = [observation, eye(sizee,1)];

number_of_nodes = [100, 100, 3];
number_of_labels = 3;
number_of_layers = length(number_of_nodes);

% unit_outputs = struct;
% for laynum = 1:number_of_layers
%     if laynum ~= number_of_layers
%         unit_outputs(laynum).a = zeros(number_of_nodes(laynum),1);
%         %to multiply with bias 
%         unit_outputs(laynum).a(end) = 1;
%     else
%         unit_outputs(laynum).a = zeros(number_of_labels,1);
%     end
% end

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
%predict 3D position
predictions = zeros(num_of_bins - 15,3);
for time_bin = 16:num_of_bins
    for layer_num = 1:number_of_layers  
        % gotta loop over layers, for each iteration of a loop over examples
        if layer_num == 1
            %first layer
            layer_output = sum(observation(1:time_bin,:))*weights{layer_num};
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
    
%     for layer_num = 1:number_of_layers  
%         % gotta loop over layers, for each iteration of a loop over examples
%         if layer_num == 1
%             %first layer
%             layer_output = sum(observation(1:time_bin,:))*weights(layer_num).a;
%             unit_outputs(layer_num).a(1:end-1) = layer_output';
%             unit_outputs(layer_num).a(1:end-1) = sigmoid(unit_outputs(layer_num).a(1:end-1));
%         elseif layer_num == number_of_layers
%             %compute final layer activation
%             layer_output = unit_outputs(layer_num-1).a'*weights(layer_num).a;
%             unit_outputs(layer_num).a = layer_output';
%             %dont compute anything using softmax, cuz use regression
%             %compute the last layer output using softmax
% %             unit_outputs(layer_num).a = softmax(unit_outputs(layer_num).a);
%         else
%             %compute hidden layer activation
%             layer_output = unit_outputs(layer_num-1).a'*weights(layer_num).a;
%             unit_outputs(layer_num).a(1:end-1) = layer_output';
%             unit_outputs(layer_num).a(1:end-1) = sigmoid(unit_outputs(layer_num).a(1:end-1));
%         end
%     end
% [~, prediction] = max(unit_outputs(number_of_layers).a);
% predictions(num_data) = prediction;
predictions(time_bin-15,:) = unit_outputs{number_of_layers}';
end

%will have 1 less interpolations than points
n = 100;
t = linspace(0,1,n)';
interpolations = cell(1,size(predictions,1)-1);
for i = 1:numel(interpolations)
    interpolations{i} = (1-t)*predictions(i,:) + t*predictions(i+1,:);
end
single_line = [];
for i = 1:numel(interpolations)
    single_line = [single_line; interpolations{i}];
end

%compute RMSE


function hidden_unit_out = sigmoid(x)
    alpha = 2.8;
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