% load data
A = importdata('train.txt');
B = importdata('test.txt');
[numTrainSamples, numTrainDim] = size(A);
[numTestSamples, numTestDim] = size(B);

% x_train = A(:,1:numTrainDim-1);
% x_test = B(:,1:numTestDim-1);
y_train = A(:,numTrainDim);
x_test_id = B(:,numTestDim);
x_train1 = A(:,1:58);
x_train2 = A(:,59:numTrainDim-1);
% x_train1 = [x_train1(:,1:8) x_train1(:,10:12) x_train1(:,15:32)];
% x_train2 = [x_train2(:,1:8) x_train2(:,10:12) x_train2(:,15:32)];
x_train3 = zeros(numTrainSamples,size(x_train1,2));
for i = 1:numTrainSamples
    for j = 1:size(x_train1,2)
        x_train3(i,j) = x_train1(i,j) * x_train2(i,j);
    end
end

x_train = [x_train1 x_train2 x_train3];

x_test1 = B(:,1:58);
x_test2 = B(:,59:numTestDim-1);
% x_test1 = [x_test1(:,1:8) x_test1(:,10:12) x_test1(:,15:32)];
% x_test2 = [x_test2(:,1:8) x_test2(:,10:12) x_test2(:,15:32)];
x_test3 = zeros(numTestSamples,size(x_test1,2));
for i = 1:numTestSamples
    for j = 1:size(x_test1,2)
        x_test3(i,j) = x_test1(i,j) * x_test2(i,j);
    end
end
x_test = [x_test1 x_test2 x_test3];
%disp(size(x_test,2));


% first normalize all the data so that it lies between -1 and 1.
%[Xtrain_norm, Xtest_norm] = normalizeAll(x_train, x_test);
%[ytrain, ytest] = normalize(ytrain, ytest);
Xtrain_norm = x_train;
Xtest_norm = x_test;

Xtrain_norm = generate_poly_features(Xtrain_norm,2);
Xtest_norm = generate_poly_features(Xtest_norm,2);
dev = Xtrain_norm(300001:numTrainSamples,:);
Xtrain_norm = Xtrain_norm(1:300000,:);
dev_labels = y_train(300001:numTrainSamples,:);
y_train = y_train(1:300000,:);
[~,numCol] = size(Xtrain_norm);
%[w,w_0] = train_ls(Xtrain_norm,y_train,1);
[w,w_0] = train_ls(Xtrain_norm,y_train,1);
col = ones(numTrainSamples-300000,1);
X_temp = [dev col];
w(numCol + 1) = w_0; 
predictY = X_temp*w;
[X,Y,T,AUC] = perfcurve(dev_labels,predictY,1);
disp(AUC);

col = ones(numTestSamples,1);
X_test = [Xtest_norm col];
predictTest = X_temp*w;

fileID = fopen('result.txt','w');
fprintf(fileID,'Id,Prediction\n');
fprintf(fileID, '%d,%1.4f\n', [transpose(x_test_id);transpose(predictTest)]);
fclose(fileID);