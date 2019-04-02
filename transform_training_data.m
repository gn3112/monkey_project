% Functions that take the "trial" structure as input and returns a
% matrix X with the training data (total number of time periods looked
% at x 98*surrounding bins) and the corresponding output Y (Change in 
% position for that timestep).
function [X, Y]  = transform_training_data(trainingData)

    X = [];
    y = [];

    % Binned data, same structure as trial but now in 20 ms bins
    binned_X = {};
    binned_Y = {};
    bin_size= 20;

    % Bin all neural activity and positions in 20ms bins:
    % (Creates new cell arrays same structure as "trial")
    i = 1;
    for trials = 1:size(trainingData,1)

        for movement = 1:size(trainingData,2)

            % Number of bins for this specific trial:
            n_bins = floor(size(trainingData(trials,movement).spikes,2)/20);

            % Make sure the last bin is handled correctly:
            if sum(20*ones(1,n_bins)) < size(trainingData(trials,movement).spikes,2)            
                last_bin = size(trainingData(trials,movement).spikes,2)-sum(20*ones(1,n_bins));
            elseif sum(20*ones(1,n_bins)) > size(trainingData(trials,movement).spikes,2)
                last_bin = sum(20*ones(1,n_bins))-size(trainingData(trials,movement).spikes,2);
            else
                last_bin = [];
            end

            % Create cell matrix with 20 data points in each cell:
            binned_spikes = mat2cell(trainingData(trials,movement).spikes, 98, [20*ones(1,n_bins),last_bin]);
            binned_X{trials,movement} =  binned_spikes;

            binned_pos = mat2cell(trainingData(trials,movement).handPos, 3, [20*ones(1,n_bins), last_bin]);
            binned_Y{trials,movement} = binned_pos;

            % Calculate spike rate in every bin and storing only last position
            % for every 20ms:
            for bin = 1:size(binned_X{trials, movement}, 2)

                % Spike Rate:
                binned_X{trials, movement}{1, bin} = sum(binned_X{trials, movement}{1, bin},2)/bin_size;

                % Storing only last position so that we can calculate change in
                % position:            
                binned_Y{trials, movement}{1, bin} =  [binned_Y{trials, movement}{1, bin}(1,end);binned_Y{trials, movement}{1, bin}(2,end);binned_Y{trials, movement}{1, bin}(2,end)];
                % Then calculating change in position:
                if bin == 1
                    binned_dY{trials, movement}{1, bin} = 0*binned_Y{trials, movement}{1, bin};
                else
                    binned_dY{trials, movement}{1, bin} =  binned_Y{trials, movement}{1, bin} - binned_Y{trials, movement}{1, bin-1};                
                end          
            end

            % Now the X and Y matrices can be constructed. Extracting one
            % 300ms period (15x20ms bins) at a time.        
            % Iterating over every 20ms period:
            for period = 1:size(binned_X{trials, movement}, 2) - 16
                   X(i,:) = reshape(transpose(cell2mat(binned_X{trials, movement}(period:15+period))),[1,98*16]);
                   Y(i,:) = transpose(cell2mat(binned_dY{trials, movement}(:,15+period)));
                   i = i+1;               
            end 
        end
    end
end
