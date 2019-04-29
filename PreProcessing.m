function [X,Y,mean_X,std_X] = PreProcessing(train_data)
%The string can either be 'equal' or 'nequal' to choose wether or not to have
%an equal number of samples for each class. In this function the data is
%split in 3 sets (training,validation,testing). Additionnaly the data is
%normalized according to the training parameters.

%         bin_l = 20;
%         targets_all = [];
%         features_all = [];
%         trial_n = 0;
%         data = train_data;
%         for mov_dir = 1:size(data,2)
%
%             for trial = 1:size(data,1)
%                 data_struc = data(trial,mov_dir);
%                 features = data_struc.spikes;
%                 targets =  data_struc.handP
%                 trial_n = trial_n + 1;
%                 features = features(:,1:end);
%                 targets = targets(:,1:end);
%                 for n_bin = 1:floor(((size(targets,2))/bin_l))
%                     features_b = features(:,(n_bin-1)*bin_l + 1:n_bin*bin_l);
%                     targets_b = targets(:,(n_bin-1)*bin_l + 1:n_bin*bin_l);
%                     features_b = sum(features_b,2);
%                     targets_b = mean(targets_b,2);
% %                     targets_b = cat(2,targets_b',[mov_dir,trial_n]);
%                     features_all = cat(1,features_all,features_b'); % 8 x trial x n_bin
%                     targets_all = cat(1,targets_all,targets_b');
%                 end
%             end
%         end

%         targets_all = targets_all(:,1:2);
%         numDatapnts = size(features_all,1); %Total number of samples used for training/validation/testing


[X, Y_all] = transform_training_data(train_data);

Y = Y_all(:,1:2);
Y(:,3) = Y_all(:,4); % direction no
numDatapnts = size(Y,1);

s = RandStream('mt19937ar','Seed',1); %Fix a seed
RandStream.setGlobalStream(s)
elems = randperm(numDatapnts);

setsData_1 = elems;

%Training/Validation data to be split
X = X(setsData_1,:);
Y = Y(setsData_1,:);

%Calculate mean and standard deviation to normalize the data
mean_X = mean(X,1);
std_X = std(X);
X = X - mean_X;
X = X./std_X;
end
