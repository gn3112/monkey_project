neural_data_cell = cell(1, size(trial,1));
positions_cell = cell(1, size(trial,1));
k = 1;
for i = 1:size(trial,2)
    for j = 1:size(trial,1)
        % each entry is a 98 by 600+ matrix
        % make st features are columns
        neural_data_cell{k} = trial(j,i).spikes';
        positions_cell{k} = trial(j,i).handPos';
        k = k + 1;
    end
end

num_fields = numel(neural_data_cell);
permutation_array = randperm(num_fields);
neural_data_cell = neural_data_cell(permutation_array);
positions_cell = positions_cell(permutation_array);
%[~,previous_order]=sort(ii)
%b=a(ii)