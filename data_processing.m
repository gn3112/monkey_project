clear, clc
data = load('monkeydata_training.mat');
data = data.trial;

bin_l = 20;
targets_all = [];
features_all = [];
trial_n = 0;

for mov_dir = 1:size(data,2)
    
    for trial = 1:size(data,1)
        data_struc = data(trial,mov_dir);
        features = data_struc.spikes;
        targets =  data_struc.handPos;
        trial_n = trial_n + 1;
        for n_bin = 1:floor((size(targets,2)/bin_l))
            features_b = features(:,(n_bin-1)*bin_l + 1:n_bin*bin_l);
            targets_b = targets(:,(n_bin-1)*bin_l + 1:n_bin*bin_l);
            features_b = mean(features_b,2);
            targets_b = mean(targets_b,2);
            targets_b = cat(2,targets_b',[mov_dir,trial_n]);
            
            
            features_all = cat(1,features_all,features_b'); % 8 x trial x n_bin
            targets_all = cat(1,targets_all,targets_b);            
        end 
    end
end

csvwrite('monkey_features.csv', features_all);
csvwrite('monkey_targets.csv', targets_all);

