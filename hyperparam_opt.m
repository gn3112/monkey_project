%% Hyperparameter optimization
clear;
clc;
tic

load('monkeydata_training.mat');
% Set random number generator
rng(2013);
ix = randperm(length(trial));

% addpath(teamName);

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:10),:);
validData = trial(ix(11:15),:);

[X,Y,mean_X,std_X] = PreProcessing(trainingData);

X_size = size(X,2); 

%Hyperparameters to be chosen: 
%Learning rate, regularization factor, std for weight init, number of
%hidden layers, number of neuron per layer, batch size or number of epoch,

Neurons = [30 50 80 100];
Layers_ = [2,3,4,5]; %for 50 neurons per layer
ep = [1,3,5,7];
Learning = [1e-4,1e-3,1e-2,1e-1];
Regularization_ = [1e-5,1e-4,1e-3,1e-2];

std_weight = [0.01,0.05,0.1,0.2];
batch_size = [50,80,100,150];

Loss = containers.Map('UniformValues',false);
accuracy_training = containers.Map('UniformValues',false);
accuracy_val = containers.Map('UniformValues',false);


%Determine how many neuron per layer assuming we have an uniform number of
%neuron per layer
count = 0;
for a = 1:length(ep)
    epoch = ep(1,a);
    for b = 1:length(Layers_)
        Layers = Layers_(1,b);
        for c = 1:length(Neurons)
            n_f = Neurons(1,c);
            N_Layer = {[n_f 2],[n_f n_f 2],[n_f n_f n_f 2],[n_f n_f n_f n_f 2]}; %30 50 80 100
            NN = N_Layer{Layers-1};
            for d = 1:length(Regularization_)
                Regularization = Regularization_(1,d);
                for e = 1:length(Learning)
                    Learning_rate = Learning(1,e);
                    algo = 'adam';
                    Std_weight = 0.12;
                    patience = 30;
                    batchsize = 120;
                    
                    [param,W,B,Ad] = initialization(Layers,NN,X_size,Learning_rate,...
                        Regularization,Std_weight,patience,epoch,batchsize); %(Number of layer, Number of neuron per layer)
                    [Loss,W,B] = training(X,Y,param,W,B,'adam',Ad);
                    count = count + 1;
                    disp(count)
                    
                    parameters = struct();
                    parameters.W = W;
                    parameters.B = B;
                    parameters.Number_of_layer = param.Number_of_layer;
                    parameters.mean_X = mean_X;
                    parameters.std_X = std_X;
                    RMSE = valid_func(validData, parameters);
                    
                    LossS{count} = Loss;
                    accuracy_valS{count} = RMSE;
                    HyperPara{count} = {epoch,Layers,n_f,Regularization,Learning_rate};
                end
            end
        end
    end
end


%% Hyperparameter optimization results processing

%Search for hyperparameter giving 99% above accuracy 
count = 0;
for k = 1:length(HyperPara)
    valid = accuracy_valS{k};
    valid = valid(:,length(valid));
    all_v(:,k) = valid;
    if valid >= 99
        count = count + 1;
        index(:,count) = k;
    end
end
max_v = max(all_v(:,index)) %Max accuracy 
index_max = find(all_v == max_v)
HyperPara{index_max} %Hyperparameters for max accuracy

Param = [1,3,5,7; %epoch
    2,3,4,5; %layers
    30 50 80 100; %neuron
    1e-5,1e-4,1e-3,1e-2; %reg
    1e-4,1e-3,1e-2,1e-1]; %learning
sum_v = zeros(5,4);

Param_s = {'1','3','5','7'; %epoch
    '2','3','4','5'; %layers
    '30' '50' '80' '100'; %neuron
    '1e-5','1e-4','1e-3','1e-2'; %reg
    '1e-4','1e-3','1e-2','1e-1'};

for k = 1:5
    P = Param(k,:);
    for i = 1:4
        P_c = P(:,i);
        count = 0;
        clear index_all
        for H_l = 1:length(HyperPara)
            HP = HyperPara{H_l};
            HP = cell2mat(HP);
            index_ = find(HP(:,k)==P_c);
            
            if isempty(index_)
                continue
            else
                count = count + 1;
                index_all(:,count) = H_l;
            end
        end
        
        for j = 1:length(index_all)
            v = accuracy_valS{index_all(:,j)};
            v_max = v(:,length(v));
            sum_v(k,i) = sum_v(k,i) + v_max;
        end
        sum_v(k,i) = sum_v(k,i)/length(index_all);
    end
    
end


%Bar chart for individual of each hyperpara in each category
hB=bar(sum_v);          % use a meaningful variable for a handle array...
hAx=gca;            % get a variable for the current axes handle
hAx.XTickLabel={'Epoch','Layers','Neuron','Regularization','Learning Rate'}; % label the ticks
hT=[];              % placeholder for text object handles

for i=1:length(hB)  % iterate over number of bar objects
    hT=[hT text(hB(i).XData+hB(i).XOffset-0.01,hB(i).YData,Param_s(:,i), ...
        'VerticalAlignment','bottom','horizontalalign','center')];
    
end

%Plot max accuracy val adn test 
figure
for l = 1:length(index_max)
    val = accuracy_valS{index_max(:,l)};
    tr = accuracy_trainingS{index_max(:,l)};
    plot(1:20:20*length(val),val,'LineWidth',2)
    hold on
    plot(1:length(tr),tr)
    hold on
    plot([1 1100],[99 99],'k','LineWidth',1)
    axis([0 850 90 101])
end

%Display all hyperparameters above 99%
for l = 1:length(index)
    val = accuracy_valS{index(:,l)};
    val = val(:,length(val))
    tr = accuracy_trainingS{index(:,l)};
end
HyperPara{index}

%Validation and loss for best parameters
figure
line(1:length(accuracy_val),accuracy_val,'Color','r','LineWidth',2)
line(1:length(accuracy_training),accuracy_training)

ax1 = gca;
ax1.XColor = 'k';
ax1.YColor = 'k';
ax1_pos = ax1.Position;
ax2 = axes('Position',ax1_pos,...
            'YAxisLocation','right',...
            'Color','none');
        
line(1:length(Loss),Loss,'Parent',ax2,'Color','b')