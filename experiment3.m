A = importdata('train.txt');
B = importdata('test.txt');
[numTrainSamples, numTrainDim] = size(A);
[numTestSamples, numTestDim] = size(B);
x_test_id = B(:,numTestDim);
y_train = A(:,numTrainDim);
neg_rows = find(y_train == -1);
y_train(neg_rows,1) = 0;
x_train = A(:,1:numTrainDim-1);
x_test = B(:,1:numTestDim-1);
[x_train, x_test] = normalizeAll(x_train, x_test);
a = x_train(:,1:58);
b = x_train(:,59:size(x_train,2));
c = x_test(:,1:58);
d = x_test(:,59:size(x_test,2));
e = a .* b;
f = c .* d;
x_train = [x_train power(x_train,2) e power(x_train,3) power(e,2) (a .* e) (b .* e) power(x_train,4) e .* power(a,2) e .* power(b,2) power(x_train,5) e .* power(a,3) a.* power(e,2) b .* power(e,2) e .* power(b,3) e .* power(a,4) e .* power(b,4) power(x_train,6) power(a,2) .* power(e,2) power(e,3) power(b,2) .* power(e,2) power(a,4) .* e power(b,4) .* e  power(x_train,7) power(a,6) .* b power(a,5) .* power(b,2) power(b,6) .* a power(x_train,8)];
x_test = [x_test power(x_test,2) f power(x_test,3) power(f,2) (c .* f) (d .* f) power(x_test,4) f .* power(c,2) f .* power(d,2) power(x_test,5) f .* power(c,3) c .* power(f,2) d .* power(f,2) f .* power(d,3) f .* power(c,4) f .* power(d,4) power(x_test,6) power(c,2) .* power(f,2)  power(f,3) power(d,2) .* power(f,2) power(c,4) .* f power(d,4) .* f power(x_test,7) power(c,6) .* d power(c,5) .* power(d,2) power(d,6) .* c power(x_test,8)];

dev = x_train(300001:numTrainSamples,:);
%x_train = x_train(1:300000,:);
dev_labels = y_train(300001:numTrainSamples,:);
%y_train = y_train(1:300000,:);

x_train = [ones(length(x_train),1) x_train];
% Initialize fitting parameters
theta = randn(size(x_train,2), 1) / sqrt(size(x_train,2));
dev = [ones(length(dev),1) dev];
% Compute and display initial cost and gradient
for i = 1:2000
    [cost, grad] = costFunction(theta, x_train, y_train);
    theta = theta - 0.1 * grad;
    if mod(i,100) == 0
        disp('cost:');
        disp(cost);
        predictY = predict(theta, dev);
        disp('i:');
        disp(i);
        [X,Y,T,AUC] = perfcurve(dev_labels,predictY,1);
        disp('AUC:');
        disp(AUC);
    end
end

x_test = [ones(length(x_test),1) x_test];
predictTest = predict(theta, x_test);

fileID = fopen('result.txt','w');
fprintf(fileID,'Id,Prediction\n');
fprintf(fileID, '%d,%1.4f\n', [transpose(x_test_id);transpose(predictTest)]);
fclose(fileID);

function [J, grad] = costFunction(theta, X, y)
m = length(y); % number of training examples
grad = zeros(size(theta));

h = sigmoid(X * theta);
J = -(1 / m) * sum( (y .* log(h)) + ((1 - y) .* log(1 - h)) );
%disp(J);

for i = 1 : size(theta, 1)
    grad(i) = (1 / m) * sum( (h - y) .* X(:, i) );
end

end

function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end

function p = predict(theta, X)
    p = sigmoid(X * theta);
end