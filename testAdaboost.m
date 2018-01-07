clear all;
close all;
clc;

A = importdata('train.txt');
B = importdata('test.txt');
[numTrainSamples, numTrainDim] = size(A);
[numTestSamples, numTestDim] = size(B);
x_test_id = B(:,numTestDim);
y_train = A(:,numTrainDim);
x_train = A(:,1:numTrainDim-1);
x_test = B(:,1:numTestDim-1);
a = x_train(:,1:58);
b = x_train(:,59:size(x_train,2));
c = x_test(:,1:58);
d = x_test(:,59:size(x_test,2));
e = a .* b;
f = c .* d;


%x_train = [x_train e];%power(x_train,2) e power(x_train,3) power(e,2) (a .* e) (b .* e) power(x_train,4) e .* power(a,2) e .* power(b,2) power(x_train,5) e .* power(a,3) a.* power(e,2) b .* power(e,2) e .* power(b,3) e .* power(a,4) e .* power(b,4) power(x_train,6) power(a,2) .* power(e,2) power(e,3) power(b,2) .* power(e,2) power(a,4) .* e power(b,4) .* e  power(x_train,7) power(a,6) .* b power(a,5) .* power(b,2) power(b,6) .* a power(x_train,8)];
%x_test = [x_test f];%power(x_test,2) f power(x_test,3) power(f,2) (c .* f) (d .* f) power(x_test,4) f .* power(c,2) f .* power(d,2) power(x_test,5) f .* power(c,3) c .* power(f,2) d .* power(f,2) f .* power(d,3) f .* power(c,4) f .* power(d,4) power(x_test,6) power(c,2) .* power(f,2)  power(f,3) power(d,2) .* power(f,2) power(c,4) .* f power(d,4) .* f power(x_test,7) power(c,6) .* d power(c,5) .* power(d,2) power(d,6) .* c power(x_test,8)];

% Keep 60% of the data for the training dataset
Xtest = x_train(300001:numTrainSamples,:);
%x_train = x_train(1:300000,:);
Ytest = y_train(300001:numTrainSamples,:);
%y_train = y_train(1:300000,:);

% Run AdaBoost
nbIterations = 1000;
fprintf('Performing %d iterations using decision stumps\n', nbIterations);
tic
[classifiers, classifiersWeights] = adaBoostTrain(x_train, y_train, nbIterations);
toc
% Get predictions on the test set
preds = adaBoostPredict(Xtest, classifiers, classifiersWeights);
[~,~,~,AUC] = perfcurve(Ytest,preds,1);
% Display the AUC
fprintf('AUC %f .\n', AUC);

predictTest = adaBoostPredict(x_test, classifiers, classifiersWeights);
fileID = fopen('result.txt','w');
fprintf(fileID,'Id,Prediction\n');
fprintf(fileID, '%d,%1.4f\n', [transpose(x_test_id);transpose(predictTest)]);
fclose(fileID);