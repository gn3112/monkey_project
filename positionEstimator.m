function [decodedPosX, decodedPosY, error] = positionEstimator(input, parameters,y_pred)

    % K-nearest
    % num_test = size(parameters.input,1);
    %
    % Ypred = zeros(num_test,1);
    %
    % for i = 1:num_test
    %     distances = sum(abs(parameters.input-input(i,:)),2);
    %     min_index = distances == min(distances);
    %     Ypred(i,:) = parameters.label(min_index);
    % end
    %
    % label = Ypred;
    if isempty(input.decodedHandPos)
        prev_pos =  input.startHandPos;
    else
        prev_pos = input.decodedHandPos(:,end);
    end
       
    test_size = size(input.spikes,2);
    input = transform_test_data(input.spikes(:,test_size-319:end));
    %function to calculate error between input and label for given weight and
    %bias. Returns the error between predicted label and true label.
    X = input;

    W = parameters.W;
    B = parameters.B;
    N_l = parameters.Number_of_layer;
    X = X - parameters.mean_X; %Normalize
    X = X./parameters.std_X;
    for N = 1:(N_l-1)
        w = strcat('w', num2str(N));
        b = strcat('b', num2str(N));

        Z = X * W(w) + B(b); % X = a where a0 is the training set
        a_to_be = Z;
        a_to_be(a_to_be<=0 ) = 0;
        A = a_to_be;

        X = A; %for next iteration
    end

    %Output layer
    w = strcat('w', num2str(N+1));
    b = strcat('b', num2str(N+1));

    Z = X * W(w) + B(b); %N+1 is the last layer
    decodedPosX = prev_pos(1,:) +  Z(:,1);
    decodedPosY = prev_pos(2,:) + Z(:,2);
%     decodedPosX = Z(:,1);
%     decodedPosY = Z(:,2);
    %         inter = exp(Z);
    %         A = inter./sum(inter,2);
    %         [p,label] = max(A,[],2);

    %         error = label - ypred;
    %         error = (1-length(error(error ~= 0))/length(label))*100;
    %         disp(error)
    
    %         %confusion matrix %comment this when training
    %         Conf = zeros(5,5);
    %         for L = 1:length(ypred)
    %             Conf(ypred(L),label(L)) = 1 + Conf(ypred(L),label(L));
    %         end
    %         % disp(Conf)
    
    % error = Test_Y - ypred;
    % error = (1-length(error(error ~= 0))/length(Test_Y))*100;
    % disp(error)
%     w = strcat('w', num2str(N+2));
%     b = strcat('b', num2str(N+2));
%     
%     Z = X * W(w) + B(b); % X = a where a0 is the training set
%     a_to_be = Z;
%     a_to_be(a_to_be<=0 ) = 0; %ReLU function
%     A = a_to_be;
%     
%     X = A; %for next iteration
    
    
    w = strcat('w', num2str(N+2));
    b = strcat('b', num2str(N+2));
    Z = X * W(w) + B(b);
    inter = exp(Z);
    A = inter./sum(inter,2);
    [p,label] = max(A,[],2);
    
    error = label - y_pred;
    error = (1-length(error(error ~= 0))/length(label))*100;
end
