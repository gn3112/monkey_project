function [dW,dB] = backward_fnc(X,Y,A,B,W,param)
%Backpropagation function to calculate the gradient.

N_l = param.Number_of_layer;
dW = containers.Map('UniformValues',false);
dB = containers.Map('UniformValues',false);

% Regression
dw = strcat('dw', num2str(N_l));
db = strcat('db', num2str(N_l));
a = strcat('a',num2str(N_l));
a_p = strcat('a',num2str(N_l-1));
w_r = strcat('w',num2str(N_l));
b_r = strcat('b',num2str(N_l));

delta_k_reg = A(a); %f(z) for last layer
delta_k_reg = 2.*(delta_k_reg-Y(:,1:2)); %(y_hat-y)
dW(dw) = transpose(A(a_p)) * delta_k_reg + param.reg*W(w_r);
dB(db) = sum(delta_k_reg,1) + param.reg*B(b_r);

% Classification
dw = strcat('dw', num2str(N_l+1));
db = strcat('db', num2str(N_l+1));
a = strcat('a',num2str(N_l+1));
a_p = strcat('a',num2str(N_l-1));
w_r = strcat('w',num2str(N_l+1));
b_r = strcat('b',num2str(N_l+1));

delta_k_class = A(a);
delta_k_class(sub2ind(size(delta_k_class),(1:numel(Y(:,end)))',Y(:,end))) = delta_k_class(sub2ind(size(delta_k_class),(1:numel(Y(:,end)))',Y(:,end))) -1;
delta_k_class = delta_k_class/length(Y);
dW(dw) = transpose(A(a_p)) * delta_k_class + param.reg*W(w_r);
dB(db) = sum(delta_k_class,1) + param.reg*B(b_r);

% dw = strcat('dw', num2str(N_l+1));
% db = strcat('db', num2str(N_l+1));
% a = strcat('a',num2str(N_l+1));
% a_p = strcat('a',num2str(N_l-1));
% w = strcat('w',num2str(N_l+2));
% w_r = strcat('w',num2str(N_l+1));
% b_r = strcat('b',num2str(N_l+1));
% 
% delta = delta_k_class * transpose(W(w));
% 
% delta(A(a)<=0) = 0;
% dW(dw) = transpose(A(a_p)) * delta + param.reg*W(w_r);
% dB(db) = sum(delta,1) + param.reg*B(b_r);
% 
% delta_k_class = delta;

k = N_l - 1;

for N = 1:(N_l-2)
    dw = strcat('dw', num2str(k));
    db = strcat('db', num2str(k));
    a = strcat('a',num2str(k));
    a_p = strcat('a',num2str(k-1));
    w = strcat('w',num2str(k+1));
    w_class = strcat('w',num2str(k+2));
    w_r = strcat('w',num2str(k));
    b_r = strcat('b',num2str(k));
    
    if N == 1
        delta = param.lambda_r * delta_k_reg * transpose(W(w)) + param.lambda_c * delta_k_class * transpose(W(w_class));
    else
        delta = delta_k * transpose(W(w));
    end
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