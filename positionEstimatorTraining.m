function  [parameters]= positionEstimatorTraining(train_data)
[X,Y,mean_X,std_X] = PreProcessing(train_data);
X_size = size(X,2);
[param,W,B,Ad] = initialization(3,[400 400 2],X_size,1e-3,...
    1e-4,0.12,40,10,128); %(Number_of_layer,Neuron_layer,X_size,Learning_rate,...
%Regularization,Std_weight,patience,epoch,batchsize) (3,[60 60 5],X_size,1e-3,...
%1e-3,0.12,40,38,100)

[Loss,W,B] = training(X,Y,param,W,B,'adam',Ad);

parameters = struct();
parameters.W = W;
parameters.B = B;
parameters.Number_of_layer = param.Number_of_layer;
parameters.mean_X = mean_X;
parameters.std_X = std_X;

    function [X,Y,mean_X,std_X] = PreProcessing(train_data)
        %The string can either be 'equal' or 'nequal' to choose wether or not to have
        %an equal number of samples for each class. In this function the data is
        %split in 3 sets (training,validation,testing). Additionnaly the data is
        %normalized according to the training parameters.
        bin_l = 20;
        targets_all = [];
        features_all = [];
        trial_n = 0;
        data = train_data;
        for mov_dir = 1:size(data,2)

            for trial = 1:size(data,1)
                data_struc = data(trial,mov_dir);
                features = data_struc.spikes;
                targets =  data_struc.handPos;
                trial_n = trial_n + 1;
                features = features(:,1:end);
                targets = targets(:,1:end);
                for n_bin = 1:floor(((size(targets,2))/bin_l)) 
                    features_b = features(:,(n_bin-1)*bin_l + 1:n_bin*bin_l);
                    targets_b = targets(:,(n_bin-1)*bin_l + 1:n_bin*bin_l);
                    features_b = sum(features_b,2);
                    targets_b = mean(targets_b,2);
%                     targets_b = cat(2,targets_b',[mov_dir,trial_n]);
                    features_all = cat(1,features_all,features_b'); % 8 x trial x n_bin
                    targets_all = cat(1,targets_all,targets_b');            
                end 
            end
        end
        targets_all = targets_all(:,1:2);
        numDatapnts = size(features_all,1); %Total number of samples used for training/validation/testing
        
        s = RandStream('mt19937ar','Seed',1); %Fix a seed
        RandStream.setGlobalStream(s)
        elems = randperm(numDatapnts);
        
        setsData_1 = elems;
        
        %Training/Validation data to be split
        X = features_all(setsData_1,:);
        Y = targets_all(setsData_1,:);
        
        %Calculate mean and standard deviation to normalize the data
        mean_X = mean(X,1);
        std_X = std(X);
        X = X - mean_X;
        X = X./std_X;
    end

    function [param,W,B,Ad] = initialization(Number_of_layer,Neuron_layer,X_size,Learning_rate,...
            Regularization,Std_weight,patience,epoch,batchsize)
        %Hyperparameters are chosen in this function and the memory is allocated
        %for the weights,bias and adam parameters matrices.  containers.Map is used
        %to store weights,biases and adam parameters. They can be accessed with a
        %key (string) and any size cell array can be contained in the same map. It
        %also makes the code more readable if string are used to access these values.
        std = Std_weight;
        param = struct();
        param.n = Learning_rate; %learning rate
        param.reg = Regularization; %regularization factor
        param.epoch = epoch;
        param.batchsize = batchsize;
        param.patience = patience;
        param.Number_of_layer = Number_of_layer;
        param.Neuron_layer = Neuron_layer;
        param.beta1 = 0.9; %Fixed hyperp
        param.beta2 = 0.999; %Fixed hyperp
        
        Ad = containers.Map('UniformValues',false);
        Row_w = X_size;
        W = containers.Map('UniformValues',false);
        B = containers.Map('UniformValues',false);
        
        for N = 1:Number_of_layer
            
            w = strcat('w', num2str(N));
            b = strcat('b', num2str(N));
            m = strcat('m',num2str(N));
            mt = strcat('mt',num2str(N));
            v = strcat('v',num2str(N));
            vt = strcat('vt',num2str(N));
            m_b = strcat('m_b',num2str(N));
            mt_b = strcat('mt_b',num2str(N));
            v_b = strcat('v_b',num2str(N));
            vt_b = strcat('vt_b',num2str(N));
            Ad(m)=0;Ad(mt)=0;Ad(v)=0;Ad(vt)=0;
            Ad(m_b)=0;Ad(mt_b)=0;Ad(v_b)=0;Ad(vt_b)=0;
            W(w) = std * randn(Row_w,Neuron_layer(N));
            B(b) = zeros(1,Neuron_layer(N));
            
            Row_w = Neuron_layer(:,N) ;
        end
    end

    function [loss,A] = forward_fnc(param,X,Y,W,B)
        %Forward pass is computed in this function. It takes as input the param
        %structure, the training input and label, weights and bias containers. It
        %returns the loss and the activation function container to be fed to the
        %backward function.
        
        %Activation functions for each hidden layer:
        %ReLU - ReLU - ... - ReLU - Nothing
        
        N_l = param.Number_of_layer;
        Neur = param.Neuron_layer;
        Z = containers.Map('UniformValues',false); %Linear function A * X + B
        A = containers.Map('UniformValues',false); % A = f(Z) where f is the activation function
        
        for N = 1:(N_l-1)
            z = strcat('z', num2str(N)); %z = w*a
            a = strcat('a', num2str(N)); %a = f(z)
            w = strcat('w', num2str(N));
            b = strcat('b', num2str(N));
            
            Z(z) = X * W(w) + B(b); % X = a where a0 is the training set
            a_to_be = Z(z);
            a_to_be(a_to_be<=0 ) = 0; %ReLU function
            A(a) = a_to_be;
            
            X = A(a); %for next iteration
        end
        
        
        %Last layer, softmax instead of ReLU
        z = strcat('z', num2str(N+1)); %z = w*a
        a = strcat('a', num2str(N+1)); %a = f(z)
        w = strcat('w', num2str(N+1));
        b = strcat('b', num2str(N+1));
        
        Z(z) = X * W(w) + B(b); %N+1 is the last layer
        
%         inter = exp(Z(z));
        A(a) = Z(z);
%         
%         y_hat = A(a);
%         tmp = y_hat(sub2ind([length(Y) Neur(:,N_l)],(1:numel(Y))',Y(:))); %find the probability of
%         %the correct class
        
        W_all = values(W);
        W_all = cellfun(@(x)x.^2,W_all,'UniformOutput',false); %square all elements of each weight matrix
        W_all = sum(cellfun(@(x) sum(x(:)),W_all)); %sum all elements of each weight matrix
        loss = mean((A(a)-Y).^2,1) + param.reg*0.5*W_all;
        disp(loss)
    end


    function [dW,dB] = backward_fnc(X,Y,A,B,W,param)
        %Backpropagation function to calculate the gradient.
        
        N_l = param.Number_of_layer;
        dW = containers.Map('UniformValues',false);
        dB = containers.Map('UniformValues',false);
        dw = strcat('dw', num2str(N_l));
        db = strcat('db', num2str(N_l));
        a = strcat('a',num2str(N_l));
        a_p = strcat('a',num2str(N_l-1));
        w_r = strcat('w',num2str(N_l));
        b_r = strcat('b',num2str(N_l));
        
        delta_k = A(a); %f(z) for last layer
        delta_k = 2.*(delta_k-Y); %(y_hat-y)
        dW(dw) = transpose(A(a_p)) * delta_k + param.reg*W(w_r);
        dB(db) = sum(delta_k,1) + param.reg*B(b_r);
        
        k = N_l - 1;
        
        for N = 1:(N_l-2)
            dw = strcat('dw', num2str(k));
            db = strcat('db', num2str(k));
            a = strcat('a',num2str(k));
            a_p = strcat('a',num2str(k-1));
            w = strcat('w',num2str(k+1));
            w_r = strcat('w',num2str(k));
            b_r = strcat('b',num2str(k));
            
            
            delta = delta_k * transpose(W(w));
            delta(A(a)<=0) = 0;
            dW(dw) = transpose(A(a_p)) * delta + param.reg*W(w_r);
            dB(db) = sum(delta,1) + param.reg*B(b_r);
            
            k = k - 1; %to store the gradient decreasingly
            delta_k = delta;
        end
        
        %First layer backprop
        dw = strcat('dw', num2str(1));
        db = strcat('db', num2str(1));
        w = strcat('w', num2str(2));
        a = strcat('a', num2str(1));
        w_r = strcat('w',num2str(1));
        b_r = strcat('b',num2str(1));
        
        delta = delta_k * transpose(W(w));
        delta(A(a) <=0 ) = 0;
        dW(dw) = transpose(X) * delta + param.reg*W(w_r);
        dB(db) = sum(delta,1) + param.reg*B(b_r);
    end

    function [Loss,W,B] = training(X,Y,param,W,B,update,Ad)
        %Training function which iterate over a specified number of epoch. Takes as
        %input the training input and label, param struct, the initial weight and
        %bias, the gradient descent method to be used (update either 'sgd' or
        %'adam'), the adam parameters and the validation set. Returns the loss,
        %accuracy for  training, weights and biases.
        %Mini-batch are used instead of feeding all the training data at each
        %forward/backward pass.
        
        %epoch = batchsize * iteration/12000; %one epoch is one full sweep through all the data
        iteration = (param.epoch*length(Y))/param.batchsize;
        X_ini = X; Y_ini = Y;
        Number_of_layer = param.Number_of_layer; %Unroll variables from structure
        
        for it = 1:iteration
            %     Mini-batch
            shuffle_indexes = randperm(size(X,1));
            shuffle_indexes = shuffle_indexes(1:param.batchsize);
            X_batch = X(shuffle_indexes, :);
            Y_batch = Y(shuffle_indexes,:);
            X(shuffle_indexes,:) = [];
            Y(shuffle_indexes,:) =  [];
            
            if size(X,1) < param.batchsize %When no batch can be extracted from
                %the total data set, start a new epoch
                X = X_ini; Y = Y_ini;
            end
            
            [loss,A] = forward_fnc(param,X_batch,Y_batch,W,B);
            [dW,dB] = backward_fnc(X_batch,Y_batch,A,B,W,param);
            
            Loss(:,it) = loss;
            
%             a = strcat('a', num2str(Number_of_layer));
%             [p,y_train] = max(A(a),[],2);
%             error_t = Y_batch - y_train;
%             accuracy_training(:,it) = (1-length(error_t(error_t ~= 0))/length(Y_batch))*100;
            
            if it == 1 %If the loss is three times the previous then the algorithm is stopped.
                continue
            elseif loss >= 8 * Loss(:,it-1)
                disp('Cost exploded')
                break
            end
            
            for ng = 1:Number_of_layer %Update the weights and biases for each layer
                w = strcat('w', num2str(ng));
                b = strcat('b', num2str(ng));
                dw = strcat('d',w);
                db = strcat('d',b);
                
                if isequal(update,'adam') %Adam
                    m = strcat('m',num2str(ng));
                    mt = strcat('mt',num2str(ng));
                    v = strcat('v',num2str(ng));
                    vt = strcat('vt',num2str(ng));
                    
                    Ad(m) = param.beta1.*Ad(m) + (1-param.beta1).*dW(dw);
                    Ad(mt) = Ad(m) ./ (1-param.beta1.^ng);
                    Ad(v) = param.beta2*Ad(v) + (1-param.beta2)*(dW(dw).^2);
                    Ad(vt) = Ad(v) / (1-param.beta2.^ng);
                    W(w) = W(w) - param.n * Ad(mt) ./ (sqrt(Ad(vt)) + 1e-8);
                    
                    m_b = strcat('m_b',num2str(ng));
                    mt_b = strcat('mt_b',num2str(ng));
                    v_b = strcat('v_b',num2str(ng));
                    vt_b = strcat('vt_b',num2str(ng));
                    
                    Ad(m_b) = param.beta1.*Ad(m_b) + (1-param.beta1)*dB(db);
                    Ad(mt_b) = Ad(m_b) ./ (1-param.beta1.^ng);
                    Ad(v_b) = param.beta2*Ad(v_b) + (1-param.beta2)*(dB(db).^2);
                    Ad(vt_b) = Ad(v_b) / (1-param.beta2.^ng);
                    B(b) = B(b) - param.n * Ad(mt_b) ./ (sqrt(Ad(vt_b)) + 1e-9);
                    
                elseif isequal(update,'sgd') %Stochastic gradient descent
                    W(w) = W(w) - param.n * dW(dw);
                    B(b) = B(b) - param.n * dB(db);
                end
                
            end
            
            if mod(it,100) == 0 %Display loss every 100 iteration
                disp(loss);
            end
        end
    end
end