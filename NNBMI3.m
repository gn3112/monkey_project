function parameters = NNBMI3(input, label, binsize)
tic
% parameters = NNBMI3(neural_data_cell(9:end),positions_cell(9:end),20)
%parameters = NNBMI3(neural_data_cell(9:end),positions_cell_2D(9:end),20)
%parameters = NNBMI3(neural_data_cell,positions_cell)

number_of_labels = size(label{1},2); % this is 3 cuz in 3D
% number of nodes in layers 1 and 2 respectively
number_of_nodes = [100, 100, number_of_labels];
number_of_layers = length(number_of_nodes);

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


% trial = 1;
l_rate = 0.0205; %0.008; 0.0205 not bad with lrate adjustment
for epoch = 1:1
    trial = 1;
for trials = 1:99
    for angles = 1:8
        l_rate = l_rate - 0.000002;
        % calculate the number of bins for each trial individually
        %start from 320ms
        trial_time = size(input{trial},1);
        adjusted_trial_time = floor(trial_time/binsize)*binsize;
        num_of_loops = (adjusted_trial_time - 300)/binsize;
        
        for loop_number = 0:num_of_loops-1
            
            example = sum(input{trial}(loop_number*binsize+1:loop_number*binsize + 300,:));
            example = example/300;
            
            if loop_number==0
                approx_derivative = ...
                    [label{trial}(loop_number*binsize+300+binsize,1),...x position
                    label{trial}(loop_number*binsize+300+binsize,2)]; % y position
            else
                approx_derivative = ...
                    [(label{trial}(loop_number*binsize+300+binsize,1) - ...
                    label{trial}(loop_number*binsize+1+300,1)),...x position
                    (label{trial}(loop_number*binsize+300+binsize,2) - ...
                    label{trial}(loop_number*binsize+1+300,2))]; % y position
            end
            
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
                else
                    %compute for hidden layers
                    layer_output = unit_outputs{layer_num-1}'*weights{layer_num};
                    unit_outputs{layer_num}(1:end-1) = layer_output';
                    unit_outputs{layer_num}(1:end-1) = sigmoid(...
                        unit_outputs{layer_num}(1:end-1));
                end
            end
            
            % need to compute the average and subtract actual position for
            % say the entire time of one trial for one angle
            % compute deltas
            for jj = number_of_layers:-1:1
                if jj == number_of_layers
                    deltas{jj} = unit_outputs{jj} - approx_derivative';
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
                                l_rate*(deltas{layer_num}(j))*example(i) + 0.00009*weights{layer_num}(i,j);% +0.001*weights(1).a(i,j)
                        end
                    end
                else
                    for i = 1:size(weights{layer_num},1)
                        for j = 1:size(weights{layer_num},2)
                            weights{layer_num}(i,j) = weights{layer_num}(i,j) - ...
                                l_rate*(deltas{layer_num}(j)*unit_outputs{layer_num-1}(i) + 0.00009*weights{layer_num}(i,j));%+ 0.001*weights(2).a(i,j)
                        end
                    end
                end
            end
            
            
        end
        % counting shuffled trials
        trial = trial+1;
    end
end
end
parameters.weights = weights;
parameters.num_of_loops = num_of_loops;
toc

function hidden_unit_out = sigmoid(x)
    alpha = 2.25;
    hidden_unit_out = 1./(1+exp(-alpha*x));
end

function diff_sigmoid = differentiate_sigmoid(x)
    diff_sigmoid = sigmoid(x)*(1 - sigmoid(x));
end

end

   