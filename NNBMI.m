function parameters = NNBMI(input, label)
tic
% input = neural_data_cell;
% label = positions_cell;
% parameters = NNBMI(neural_data_cell(9:end),positions_cell(9:end))
%parameters = NNBMI(neural_data_cell,positions_cell)

number_of_labels = size(label{1},2); % this is 3 cuz in 3D
% number of nodes in layers 1 and 2 respectively
number_of_nodes = [100, 100, number_of_labels];
number_of_layers = length(number_of_nodes);
%number_of_features_ = size(input{1},2); % 98 neural units

trial_times = zeros(1, numel(input));
for j = 1:numel(input)
        trial_times(j) = size(input{j},1);
end   
max_time = max(trial_times);


% add bias term to each trial, thats an extra feature with 1 and rest 0s
for field = 1:numel(input)
    input{field} = [input{field}, eye(size(input{field},1),1)];
end
number_of_features = size(input{field},2);

weights = cell(1, number_of_layers);
for laynum = 1:number_of_layers
    if laynum == 1
        %includes biases for first layer, also no weights to go into the last node
        %because it should remain 1
        weights{laynum} = randn(number_of_features, number_of_nodes(laynum)-1);
    elseif laynum == number_of_layers
        % last layer has as many weights as outputs
        weights{laynum} = randn(number_of_nodes(laynum-1), number_of_nodes(laynum));
    else
        %weights for other layers, weight term taken into account
        weights{laynum} = randn(number_of_nodes(laynum-1), number_of_nodes(laynum)-1);
    end
end

unit_outputs = cell(1, number_of_layers);
for laynum = 1:number_of_layers
    if laynum ~= number_of_layers
        unit_outputs{laynum} = zeros(number_of_nodes(laynum),1);
        %to multiply with bias 
        unit_outputs{laynum}(end) = 1;
    else
        unit_outputs{laynum} = zeros(number_of_labels,1);
    end
end

deltas = cell(1,number_of_layers);
for laynum = 1:number_of_layers
        deltas{laynum} = zeros(size(unit_outputs{laynum},1), 1);
end

    
% need to calculate how many time windows there are per trial based on the
% maximum time. Also fill with 0s the trials that have less times
% need to account for the fact that start predicting after the first 300 ms
num_of_bins = ceil(max_time/20);
% pad with zeros the trials that are shorter in time than num_of_bins*20
for field = 1:numel(input)
    input{field} = [input{field}; ...
        zeros(num_of_bins*20-size(input{field},1), number_of_features)];
end

trial = 1;
l_rate = 0.05;
for trials = 1:99
    for angles = 1:8
        %start from 320ms
        for time_bin = 16:num_of_bins
            %input num will be the index for data
            %this is a row vector
            % sum the entries in each colum
            example = sum(input{trial}(1:time_bin,:));
            % 3d column vector, calculating the average position over 20ms
            % time window. so below we sum them up and then divide
            actual_position = sum(label{trial}(20*time_bin: 20*(time_bin-1),:));
            actual_position = actual_position'/20;
            % propagate forward in the network
            % the first layer of input values is in the form of a row vector
            % compute values for hidden layers
            
            for layer_num = 1:number_of_layers
                % gotta loop over layers, for each iteration of a loop over examples
                if layer_num == 1
                    layer_output = example*weights{layer_num};
                    %want to leave the term for bias = 1
                    unit_outputs{layer_num}(1:end-1) = layer_output';
                    %apply sigmoid to the layer
                    unit_outputs{layer_num}(1:end-1) = sigmoid(unit_outputs{layer_num}(1:end-1));
                elseif layer_num == number_of_layers
                    %compute final layer activation and values
                    layer_output = unit_outputs{layer_num-1}'*weights{layer_num};
                    %for regression don't take any function of that
                    unit_outputs{layer_num} = layer_output';
                    %compute the last layer output using softmax
                    %unit_outputs(layer_num).a = softmax(unit_outputs(layer_num).a);
                else
                    %compute for hidden layers
                    layer_output = unit_outputs{layer_num-1}'*weights{layer_num};
                    unit_outputs{layer_num}(1:end-1) = layer_output';
                    %compute the last layer output using softmax
                    unit_outputs{layer_num}(1:end-1) = sigmoid(...
                        unit_outputs{layer_num}(1:end-1));
                end
            end
            
            
            
            % need to compute the average and subtract actual position for
            % say the entire time of one trial for one angle
            % compute deltas
            
            
            
            for jj = number_of_layers:-1:1
                if jj == number_of_layers
                    deltas{jj} = unit_outputs{jj} - actual_position;
                elseif jj == (number_of_layers-1)
                    %want to update the next to last layers using all the deltas
                    for ii = 1:size(unit_outputs{jj},1)
                        deltas{jj}(ii) =  differentiate_sigmoid(...
                            unit_outputs{jj}(ii))*(weights{jj+1}(ii,:)*deltas{jj+1});
                    end
                else
                    %for all other layers use all delta values except the last
                    %one because we want to keep the last weigh a constant value
                    for ii = 1:size(unit_outputs{jj},1)
                        deltas{jj}(ii) =  differentiate_sigmoid(...
                            unit_outputs{jj}(ii))*(weights{jj+1}(ii,:)*deltas{jj+1}(1:end-1));
                    end
                end
            end
            
            for layer_num = number_of_layers:-1:1
                %accounts for bias term because iterate over number of weights
                if layer_num == 1
                    for i = 1:size(weights{layer_num},1)
                        for j = 1:size(weights{layer_num},2)
                            weights{layer_num}(i,j) = weights{layer_num}(i,j) - ...
                                l_rate*(deltas{layer_num}(j))*example(i) + 0.0003*weights{layer_num}(i,j);% +0.001*weights(1).a(i,j)
                        end
                    end
                else
                    for i = 1:size(weights{layer_num},1)
                        for j = 1:size(weights{layer_num},2)
                            weights{layer_num}(i,j) = weights{layer_num}(i,j) - ...
                                l_rate*(deltas{layer_num}(j)*unit_outputs{layer_num-1}(i) + 0.0003*weights{layer_num}(i,j));%+ 0.001*weights(2).a(i,j)
                        end
                    end
                end
            end
            
            
        end
        % counting shuffled trials
        trial = trial+1;
    end
end
parameters.weights = weights;
parameters.num_of_bins = num_of_bins;
toc

function hidden_unit_out = sigmoid(x)
    alpha = 2.8;
    % can be applied to scalars and vectors
    hidden_unit_out = 1./(1+exp(-alpha*x));
end

function diff_sigmoid = differentiate_sigmoid(x)
    %refers to the sigmoid above
    diff_sigmoid = sigmoid(x)*(1 - sigmoid(x));
end

end



%{
function parameters = NNBMI(input, label)
tic
% start with one hidden layer of the same size as the number of labels
% weight matrix where each layer in the third dimension corresponds to a 
% layer of weights in the NN 
% For each such 2D matrix a column vector is a single weight vector same
% size as the input, number of column vectors is the number of nodes
% in the layer to which the weights lead
% input = neural_data_cell;
% label = positions_cell;
% parameters = NNBMI(neural_data_cell,positions_cell)

number_of_labels = size(label{1},2); % this is 3 cuz in 3D
% number of nodes in layers 1 and 2 respectively
number_of_nodes = [100, 100, number_of_labels];
number_of_layers = length(number_of_nodes);
%number_of_features_ = size(input{1},2); % 98 neural units

% means = zeros(1,number_of_features_);
% stds = zeros(1,number_of_features_);
% for feature = 1:number_of_features_
%     means(feature) = mean(input(:,feature));
%     stds(feature) = std(input(:,feature));
% end
% 
% for i = 1:number_of_features_
%     input(:,i) = (input(:,i) - means(i))./stds(i);
% end

trial_times = zeros(1, numel(input));
for j = 1:numel(input)
        trial_times(j) = size(input{j},1);
end   
max_time = max(trial_times);


% add bias term to each trial, thats an extra feature with 1 and rest 0s
for field = 1:numel(input)
    input{field} = [input{field}, eye(size(input{field},1),1)];
end
number_of_features = size(input{field},2);

weights = struct;
for laynum = 1:number_of_layers
    if laynum == 1
        %includes biases for first layer, also no weights to go into the last node
        %because it should remain 1
        weights(laynum).a = randn(number_of_features, number_of_nodes(laynum)-1);
    elseif laynum == number_of_layers
        % last layer has as many weights as outputs
        weights(laynum).a = randn(number_of_nodes(laynum-1), number_of_nodes(laynum));
    else
        %weights for other layers, weight term taken into account
        weights(laynum).a = randn(number_of_nodes(laynum-1), number_of_nodes(laynum)-1);
    end
end

% matrix to store unit outputs 
%need to make hidden units and outputs units separately
unit_outputs = struct;
for laynum = 1:number_of_layers
    if laynum ~= number_of_layers
        unit_outputs(laynum).a = zeros(number_of_nodes(laynum),1);
        %to multiply with bias 
        unit_outputs(laynum).a(end) = 1;
    else
        unit_outputs(laynum).a = zeros(number_of_labels,1);
    end
end

%matrix to store deltas
deltas = struct;
for laynum = 1:number_of_layers
        deltas(laynum).a = zeros(size(unit_outputs(laynum).a,1), 1);
end

    
% need to calculate how many time windows there are per trial based on the
% maximum time. Also fill with 0s the trials that have less times
% need to account for the fact that start predicting after the first 300 ms
num_of_bins = ceil(max_time/20);
% pad with zeros the trials that are shorter in time than num_of_bins*20
for field = 1:numel(input)
    input{field} = [input{field}; ...
        zeros(num_of_bins*20-size(input{field},1), number_of_features)];
end

trial = 1;
l_rate = 0.05;%/sqrt(sqrt(input_num));
for trials = 1:100
    for angles = 1:8
        %start from 320ms
        for time_bin = 16:num_of_bins
            %input num will be the index for data
            %this is a row vector
%             example = input(input_num,:);
            % sum the entries in each colum
            example = sum(input{trial}(1:time_bin,:));
            
            %actual_position = label(input_num);
            % 3d column vector
            actual_position = label{trial}(time_bin,:)';
            
            %         example_label = label(input_num);
            %         %get corresponding label for random sample
            %         binary_label = zeros(number_of_labels,1);
            %         % desired index of the label is set to 1
            %         binary_label(example_label) = 1;
            
            % propagate information forward
            % the first layer of input values is in the form of a row vector
            % compute values for hidden layers
            for layer_num = 1:number_of_layers
                % gotta loop over layers, for each iteration of a loop over examples
                if layer_num == 1
                    layer_output = example*weights(layer_num).a;
                    %want to leave the term for bias = 1
                    unit_outputs(layer_num).a(1:end-1) = layer_output';
                    %apply sigmoid to the layer
                    unit_outputs(layer_num).a(1:end-1) = sigmoid(unit_outputs(layer_num).a(1:end-1));
                elseif layer_num == number_of_layers
                    %compute final layer activation and values
                    layer_output = unit_outputs(layer_num-1).a'*weights(layer_num).a;
                    %for regression don't take any function of that
                    unit_outputs(layer_num).a = layer_output';
                    %compute the last layer output using softmax
                    %unit_outputs(layer_num).a = softmax(unit_outputs(layer_num).a);
                else
                    %compute for hidden layers
                    layer_output = unit_outputs(layer_num-1).a'*weights(layer_num).a;
                    unit_outputs(layer_num).a(1:end-1) = layer_output';
                    %compute the last layer output using softmax
                    unit_outputs(layer_num).a(1:end-1) = sigmoid(...
                        unit_outputs(layer_num).a(1:end-1));
                end
            end
            
            % need to compute the average and subtract actual position for
            % say the entire time of one trial for one angle
            % compute deltas
            for jj = number_of_layers:-1:1
                if jj == number_of_layers
                    %last layer deltas
                    %                 for ii = 1:size(unit_outputs(jj).a,1)
                    %                     deltas(jj).a(ii) = delta_last(ii, unit_outputs, actual_position);
                    %                 end
                    deltas(jj).a = unit_outputs(jj).a - actual_position;
                elseif jj == (number_of_layers-1)
                    %want to update the next to last layers using all the deltas
                    for ii = 1:size(unit_outputs(jj).a,1)
                        deltas(jj).a(ii) =  differentiate_sigmoid(...
                            unit_outputs(jj).a(ii))*(weights(jj+1).a(ii,:)*deltas(jj+1).a);
                    end
                else
                    %for all other layers use all delta values except the last
                    %one because we want to keep the last weigh a constant value
                    for ii = 1:size(unit_outputs(jj).a,1)
                        deltas(jj).a(ii) =  differentiate_sigmoid(...
                            unit_outputs(jj).a(ii))*(weights(jj+1).a(ii,:)*deltas(jj+1).a(1:end-1));
                    end
                end
            end
            
            % weight update with GD
            for layer_num = number_of_layers:-1:1
                %accounts for bias term because iterate over number of weights
                if layer_num == 1
                    for i = 1:size(weights(layer_num).a,1)
                        for j = 1:size(weights(layer_num).a,2)
                            weights(layer_num).a(i,j) = weights(1).a(i,j) - ...
                                l_rate*(deltas(layer_num).a(j))*example(i) + 0.0005*weights(layer_num).a(i,j);% +0.001*weights(1).a(i,j)
                        end
                    end
                else
                    for i = 1:size(weights(layer_num).a,1)
                        for j = 1:size(weights(layer_num).a,2)
                            weights(layer_num).a(i,j) = weights(layer_num).a(i,j) - ...
                                l_rate*(deltas(layer_num).a(j)*unit_outputs(layer_num-1).a(i) + 0.0005*weights(layer_num).a(i,j));%+ 0.001*weights(2).a(i,j)
                        end
                    end
                end
            end
        end
        % counting shuffled trials
        trial = trial+1;
    end
end
parameters.weights = weights;
parameters.num_of_bins = num_of_bins;
toc

function hidden_unit_out = sigmoid(x)
    alpha = 2.8;
    % can be applied to scalars and vectors
    hidden_unit_out = 1./(1+exp(-alpha*x));
end

function diff_sigmoid = differentiate_sigmoid(x)
    %refers to the sigmoid above
    diff_sigmoid = sigmoid(x)*(1 - sigmoid(x));
end
% 
% function last_layer_out = softmax(input)
%     last_layer_out = exp(input)/(sum(exp(input)));
% end

% function delta_last_layer = delta_last(j, unit_outp, position)
%     delta_last_layer = unit_outp(j) - position(j);
% end
%parameters = NN(partitioned_data.train_inp, partitioned_data.train_label)
end

%}
   