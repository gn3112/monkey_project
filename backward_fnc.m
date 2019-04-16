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