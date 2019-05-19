%% Initialization
%clear ; close all; clc

%% 参数设置与数据输入
input_layer_size  = 3993;  % 每个数据的特征数
hidden_layer_size = 100;   % 隐藏层中神经元数
num_labels = 3;          % 最终层（类别数）


fprintf('\n正在读入数据与进行MFCC...\n');
%load('data_test.mat');
%load('data.mat');
fprintf('\n数据读入完成，请检查，程序已经暂停...\n');
pause;
% m-数据组数  n-特征个数

% 此处：读取数据(暂定，要自写MFCC)
%load('ex4data1.mat'); % 变量X m行n列  变量y m行1列

% 此处预留: 将从wav中读取出的数据用MFCC转化
% 建议写一个函数，因为预测的时候也要调用

%% 随机初始化

fprintf('\nInitializing Neural Network Parameters ...\n')

%Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
%Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
%initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


initial_nn_params = [Theta1(:) ; Theta2(:)];
%% 梯度检验
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% 神经网络学习

fprintf('\nTraining Neural Network... \n')

% 迭代次数，可调整
options = optimset('MaxIter', 1000);

% 正则系数，可调整
lambda = 0;

% 消费函数的函数指针
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xtrain, ytrain, lambda);

% 梯度下降的轮子
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% 两个矩阵，拿出来
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% 检验模型（注意这里是拿训练的去检验）
pred1 = predict(Theta1, Theta2, Xtrain);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred1 == ytrain)) * 100);

%% 检验模型（测试集）
pred2 = predict(Theta1, Theta2, X);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred2 == y)) * 100);