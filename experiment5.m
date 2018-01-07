% loading the training set.

x_train = importdata('train.txt');
[numTrainSamples, numTrainDim] = size(x_train);

% actual lables are the last column of training set and rest are the
% features.

y_train = x_train(:,numTrainDim);
x_train = x_train(:,1:numTrainDim-1);

% separating first 58 features of the first person (a) and next 58 features 
% of the second person(b) and using all the products in the expansion
% of (a+b)^8

a = x_train(:,1:58);
b = x_train(:,59:size(x_train,2));
e = a .* b;

% adding all the products in the expansion of (a+b)^8 to the training set.

x_train = [x_train power(x_train,2) e power(x_train,3) power(e,2) (a .* e) (b .* e) power(x_train,4) e .* power(a,2) e .* power(b,2) power(x_train,5) e .* power(a,3) a .* power(e,2) b .* power(e,2) e .* power(b,3) power(x_train,6) power(a,2) .* power(e,2) power(e,3) power(b,2) .* power(e,2) power(a,4) .* e power(b,4) .* e power(x_train,7) power(a,6) .* b power(a,5) .* power(b,2) power(a,4) .* power(b,3) power(a,3) .* power(b,4) power(a,2) .* power(b,5) a .* power(b,6) power(x_train,8) power(a,7) .* b power(a,6) .* power(b,2) power(a,5) .* power(b,3) power(a,4) .* power(b,4) power(a,3) .* power(b,5) power(a,2) .* power(b,6) power(a,1) .* power(b,7)];

% clearing temporary variables a,b,c for freeing the memory.

clear a;
clear b;
clear e;

% used first 300000 samples as training set and remaining as validation
% set for experiments but final result was computed on the whole dataset.

%x_train = x_train(1:300000,:);
%y_train = y_train(1:300000,:);

% Ridge regression with lambda = 10.

[~,numCol] = size(x_train);
[w,w_0] = train_rr(x_train,y_train,10);
w(numCol + 1) = w_0;

% making dev set.

dev = x_train(300001:numTrainSamples,:);
dev_labels = y_train(300001:numTrainSamples,:);

% predicting on dev set.

col = ones(length(dev),1);
dev = [dev col];
predictY = dev*w;

% calculating AUC on dev set.

[~,~,~,AUC] = perfcurve(dev_labels,predictY,1);
disp(AUC);

% clearing all the variables for freeing memory.

clear x_train;
clear y_train;
clear dev;
clear dev_labels;
clear predictY;

% loading the test dataset

x_test = importdata('test.txt');
[numTestSamples, numTestDim] = size(x_test);

% taking last column as the user id.

x_test_id = x_test(:,numTestDim);
x_test = x_test(:,1:numTestDim-1);

% separating first 58 features of the first person (c) and next 58 features 
% of the second person(d) and using all the products in the expansion
% of (c+d)^8 as done in the training set.

c = x_test(:,1:58);
d = x_test(:,59:size(x_test,2));
f = c .* d;

% adding all the products in the expansion of (c+d)^8 to the test set.

x_test = [x_test power(x_test,2) f power(x_test,3) power(f,2) (c .* f) (d .* f) power(x_test,4) f .* power(c,2) f .* power(d,2) power(x_test,5) f .* power(c,3) c .* power(f,2) d .* power(f,2) f .* power(d,3) power(x_test,6) power(c,2) .* power(f,2)  power(f,3) power(d,2) .* power(f,2) power(c,4) .* f power(d,4) .* f power(x_test,7) power(c,6) .* d power(c,5) .* power(d,2) power(c,4) .* power(d,3) power(c,3) .* power(d,4) power(c,2) .* power(d,5) c .* power(d,6) power(x_test,8) power(c,7) .* d power(c,6) .* power(d,2) power(c,5) .* power(d,3) power(c,4) .* power(d,4) power(c,3) .* power(d,5) power(c,2) .* power(d,6) power(c,1) .* power(d,7)] ;

% clearing the variables

clear c;
clear d;
clear f;

% predicting on test set using w computed.

col = ones(numTestSamples,1);
x_test = [x_test col];
predictTest = x_test*w;

% writing to the file.

fileID = fopen('result.txt','w');
fprintf(fileID,'Id,Prediction\n');
fprintf(fileID, '%d,%1.4f\n', [transpose(x_test_id);transpose(predictTest)]);
fclose(fileID);