%% Initialization
%clear ; close all; clc

%% ������������������
input_layer_size  = 3993;  % ÿ�����ݵ�������
hidden_layer_size_1 = 100;   % ���ز�1����Ԫ��
hidden_layer_size_2 = 30;   %���ز�2����Ԫ��
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

Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size_1);
Theta2 = randInitializeWeights(hidden_layer_size_1, hidden_layer_size_2);
Theta3 = randInitializeWeights(hidden_layer_size_2, num_labels);

% Unroll parameters
%initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


initial_nn_params = [Theta1(:) ; Theta2(:) ; Theta3(:)];
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
                                   hidden_layer_size_1, ...
                                   hidden_layer_size_2, ...
                                   num_labels, Xtrain, ytrain, lambda);

% �ݶ��½�������
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% ���������ó���

unrolled_size=size(Theta1,1)*size(Theta1,2);
Theta1 = reshape(nn_params(1:unrolled_size), hidden_layer_size_1, (input_layer_size + 1));
nn_params=nn_params(1+unrolled_size:end);

unrolled_size=size(Theta2,1)*size(Theta2,2);
Theta2 = reshape(nn_params(1:unrolled_size), hidden_layer_size_2, (hidden_layer_size_1 + 1));
nn_params=nn_params(1+unrolled_size:end);

unrolled_size=size(Theta3,1)*size(Theta3,2);
Theta3 = reshape(nn_params(1:unrolled_size), num_labels, (hidden_layer_size_2 + 1));
%nn_params=nn_params(1+unrolled_size:end);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ����ģ�ͣ�ע����������ѵ����ȥ���飩
pred1 = predict(Theta1, Theta2, Theta3, Xtrain);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred1 == ytrain)) * 100);

%% ����ģ�ͣ����Լ���
pred2 = predict(Theta1, Theta2, Theta3, X);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred2 == y)) * 100);