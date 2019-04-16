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
end
