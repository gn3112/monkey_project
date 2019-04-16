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