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