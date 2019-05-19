function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size_1, ...
                                   hidden_layer_size_2, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 4 layer neural network
Theta1=zeros(hidden_layer_size_1,input_layer_size+1);
Theta2=zeros(hidden_layer_size_2,hidden_layer_size_1+1);
Theta3=zeros(num_labels,hidden_layer_size_2+1);

unrolled_size=size(Theta1,1)*size(Theta1,2);
Theta1 = reshape(nn_params(1:unrolled_size), hidden_layer_size_1, (input_layer_size + 1));
nn_params=nn_params(1+unrolled_size:end);

unrolled_size=size(Theta2,1)*size(Theta2,2);
Theta2 = reshape(nn_params(1:unrolled_size), hidden_layer_size_2, (hidden_layer_size_1 + 1));
nn_params=nn_params(1+unrolled_size:end);

unrolled_size=size(Theta3,1)*size(Theta3,2);
Theta3 = reshape(nn_params(1:unrolled_size), num_labels, (hidden_layer_size_2 + 1));
%nn_params=nn_params(1+unrolled_size:end);

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(1,m);X'];

z2 = Theta1 * a1;
a2 = [ones(1,m) ; sigmoid(z2)];
z2 = [ones(1,m);z2];

z3 = Theta2 * a2;
a3 = [ones(1,m) ; sigmoid(z3)];
z3 = [ones(1,m);z3];

z4 = Theta3 * a3;
a4 = sigmoid(z4); 
h = a4'; % (num_labels *  m)' = (m * num_labels)

yk=zeros(size(y,1),num_labels);
for i=1:size(y,1) 
    yk(i,:)=(1:num_labels)==y(i);
end;
J = -1/m * sum(sum(yk.*log(h)+(1-yk).*log(1-h)))...
    + lambda / (2*m)*(sum(sum(Theta1(:,2:end).*Theta1(:,2:end))) ...
                     +sum(sum(Theta2(:,2:end).*Theta2(:,2:end))) ...
                     +sum(sum(Theta3(:,2:end).*Theta3(:,2:end))));

delta4 = a4 - yk';
delta3 = (Theta3' * delta4) .* sigmoidGradient(z3);
delta3 = delta3(2:end,:);
delta2 = (Theta2' * delta3) .* sigmoidGradient(z2);
delta2 = delta2(2:end,:);
Theta1_grad = delta2 * a1';
Theta2_grad = delta3 * a2';
Theta3_grad = delta4 * a3';

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda * Theta1(:,2:end);
Theta1_grad = Theta1_grad./m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda * Theta2(:,2:end);
Theta2_grad = Theta2_grad./m;
Theta3_grad(:,2:end) = Theta3_grad(:,2:end) + lambda * Theta3(:,2:end);
Theta3_grad = Theta3_grad./m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end
