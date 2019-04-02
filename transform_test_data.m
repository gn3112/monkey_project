% Functions that take the "trial" structure as input and returns a
% matrix X with the training data (total number of time periods looked
% at x 98*surrounding bins) and the corresponding output Y (Change in 
% position for that timestep).
function X_test  = transform_test_data(testData)

    bin_size= 20;
    n_bins = 16;

    % Create cell matrix with 20 data points in each cell:
    binned_spikes = mat2cell(testData, 98, [20*ones(1,n_bins)]);

    % Calculate spike rate in every bin:
    for bin = 1:n_bins
        % Spike Rate:
        binned_spikes{1, bin} = sum(binned_spikes{1, bin},2)/bin_size;
    end

    % Now the X matrix can be constructed.      
    X_test = reshape(transpose(cell2mat(binned_spikes)),[1,98*16]);             

end
