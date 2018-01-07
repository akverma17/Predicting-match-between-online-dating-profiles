A = importdata('train.txt');
B = importdata('test.txt');
[numTrainSamples, numTrainDim] = size(A);
[numTestSamples, numTestDim] = size(B);
x_test_id = B(:,numTestDim);
y_train = A(:,numTrainDim);
x_train = A(:,1:numTrainDim-1);
x_test = B(:,1:numTestDim-1);
% [x_train, x_test] = normalizeAll(x_train, x_test);
x_train = [x_train power(x_train(:,1:58) .* x_train(:,59:size(x_train,2)),2)];
x_test = [x_test power(x_test(:,1:58) .* x_test(:,59:size(x_test,2)),2)];
%x_train = generate_poly_features(x_train,4);
%x_test = generate_poly_features(x_test,4);
x_train = zscore(x_train);
x_test = zscore(x_test);

dev = x_train(300001:numTrainSamples,:);
x_train = x_train(1:300000,:);
dev_labels = y_train(300001:numTrainSamples,:);
y_train = y_train(1:300000,:);

X = x_train(1:100,:);
K = power(1 + (x_train * transpose(X)), 2);
alpha = inv(transpose(K) * K) * transpose(K) * y_train;
predictY = zeros(size(dev,1),1);
for i = 1:size(dev,1)
    for j = 1:size(X,1)
        predictY(i,1) = predictY(i,1) + alpha(j,1) * power(1 + dot(dev(i,:), X(j,:)), 2);
    end
end

% [~,numCol] = size(x_train);
% [w,w_0] = train_ls(x_train,y_train,1);
% w(numCol + 1) = w_0;
% col = ones(length(dev),1);
% X_temp = [dev col];
% predictY = X_temp*w;

[X,Y,T,AUC] = perfcurve(dev_labels,predictY,1);
disp(AUC);

% col = ones(numTestSamples,1);
% X_test = [x_test col];
% predictTest = X_test*w;


% fileID = fopen('result.txt','w');
% fprintf(fileID,'Id,Prediction\n');
% fprintf(fileID, '%d,%1.4f\n', [transpose(x_test_id);transpose(predictTest)]);
% fclose(fileID);