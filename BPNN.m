%% Initialization
%clear ; close all; clc

%% ������������������
input_layer_size  = 3993;  % ÿ�����ݵ�������
hidden_layer_size = 100;   % ���ز�����Ԫ��
num_labels = 3;          % ���ղ㣨�������


fprintf('\n���ڶ������������MFCC...\n');
%load('data_test.mat');
%load('data.mat');
fprintf('\n���ݶ�����ɣ����飬�����Ѿ���ͣ...\n');
pause;
% m-��������  n-��������

% �˴�����ȡ����(�ݶ���Ҫ��дMFCC)
%load('ex4data1.mat'); % ����X m��n��  ����y m��1��

% �˴�Ԥ��: ����wav�ж�ȡ����������MFCCת��
% ����дһ����������ΪԤ���ʱ��ҲҪ����

%% �����ʼ��

fprintf('\nInitializing Neural Network Parameters ...\n')

%Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
%Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
%initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


initial_nn_params = [Theta1(:) ; Theta2(:)];
%% �ݶȼ���
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ������ѧϰ

fprintf('\nTraining Neural Network... \n')

% �����������ɵ���
options = optimset('MaxIter', 1000);

% ����ϵ�����ɵ���
lambda = 0;

% ���Ѻ����ĺ���ָ��
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xtrain, ytrain, lambda);

% �ݶ��½�������
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% ���������ó���
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ����ģ�ͣ�ע����������ѵ����ȥ���飩
pred1 = predict(Theta1, Theta2, Xtrain);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred1 == ytrain)) * 100);

%% ����ģ�ͣ����Լ���
pred2 = predict(Theta1, Theta2, X);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred2 == y)) * 100);