load('data.mat');
rate = 0.8;

rate = 1 - rate;
number = size(y,1) / 3;
k = ceil(number*rate);

training_set = randperm(number,k);
training_set(k + 1 : k * 2) = training_set(1:k) + number;
training_set(2*k + 1 : k * 3) = training_set(1:k) + 2*number;


Xtrain = X(training_set , :);
ytrain = y(training_set , :);
X(training_set,:)=[];
y(training_set,:)=[];

save('data_test.mat','X','y','Xtrain','ytrain');