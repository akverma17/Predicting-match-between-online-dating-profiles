A = importdata('train.txt');
B = importdata('test.txt');
[numTrainSamples, numTrainDim] = size(A);
[numTestSamples, numTestDim] = size(B);
x_test_id = B(:,numTestDim);
y_train = A(:,numTrainDim);
%[A, B] = normalizeAll(A, B);

x_train1 = A;%(:,1:58);
x_train2 = A;%(:,59:numTrainDim-1);
x_train1 = [x_train1(:,2:11) x_train1(:,15:17) x_train1(:,19:25) x_train1(:,52:58)];
x_train2 = [x_train2(:,60:69) x_train2(:,73:75) x_train2(:,77:83) x_train2(:,110:numTrainDim-1)];

x_train = [x_train1 x_train2];
%x_train = [x_train(:,2:4) x_train(:,11:13) x_train(:,20) x_train(:,26:27) x_train(:,29:31) x_train(:,38:40) x_train(:,47) x_train(:,53:54)];
%disp(x_train(1,:));
x_train1 = x_train(:,1:27);
x_train2 = x_train(:,28:54);
x_train = x_train1 .* x_train2;
x_train3 = zeros(numTrainSamples,378);
k = 1;
for i = 1:size(x_train,2)
    for j = 1:size(x_train,2)
    x_train3(:,k) = x_train(:,i) .* x_train(:,j);
    k = k+1;
    end
end
x_train = x_train3;%[x_train x_train3];
pos_rows = find(y_train == 1);
%disp([transpose(x_train1(pos_rows(1:5,:))) transpose(x_train2(pos_rows(1:5,:)))]); 
neg_rows = find(y_train == -1);
xpos = x_train(pos_rows,:);
xneg = x_train(neg_rows,:);
ypos = y_train(pos_rows,:);
yneg = y_train(neg_rows,:);
a = repmat(xpos,7,1);
b = repmat(ypos,7,1);
x_train=[xneg(1:200000,:);a(1:200000,:)];
y_train = [yneg(1:200000,:);b(1:200000,:)];
%neg_rows = neg_rows(1:length(pos_rows));
%rows = sort([pos_rows, neg_rows]);
%x_train = x_train(rows,:);
%y_train = y_train(rows,:);
s=sign(y_train);
ipositif=sum(s(:)==1);
inegatif=sum(s(:)==-1);
% disp(ipositif);
% disp(inegatif);

%x_train = x_train1 .* x_train2;

%x_train = [x_train power(x_train,2) power(x_train,3)];% x_train .* power(x_train,2)];

x_test1 = B;%(:,1:58);
x_test2 = B;%(:,59:numTestDim-1);
x_test1 = [x_test1(:,2:11) x_test1(:,15:17) x_test1(:,19:25) x_test1(:,52:58)];
x_test2 = [x_test2(:,60:69) x_test2(:,73:75) x_test2(:,77:83) x_test2(:,110:numTestDim-1)];
x_test = [x_test1 x_test2];
%x_test = [x_test(:,2:4) x_test(:,11:13) x_test(:,20) x_test(:,26:27) x_test(:,29:31) x_test(:,38:40) x_test(:,47) x_test(:,53:54)];
x_test1 = x_test(:,1:27);
x_test2 = x_test(:,28:54);
x_test = x_test1 .* x_test2;
%disp(x_test(1,:));
x_test3 = zeros(numTestSamples,378);
k = 1;
for i = 1:size(x_test,2)
    for j = 1:size(x_test,2)
    x_test3(:,k) = x_test(:,i) .* x_test(:,j);
    k = k+1;
    end
end
x_test = x_test3;%[x_test x_test3];
%x_test = x_test1 .* x_test2;

%x_train = generate_poly_features(x_train,2);
%x_test = generate_poly_features(x_test,2);

x_train_copy = x_train;
y_train_copy = y_train;

pos_rows = find(y_train == 1);
neg_rows = find(y_train == -1);
dev_pos_rows = pos_rows(1:5000,:);
dev_neg_rows = neg_rows(1:5000,:);
train_pos_rows = pos_rows(5001:length(pos_rows),:);
train_neg_rows = neg_rows(5001:length(neg_rows),:);

dev_rows = sort([dev_pos_rows, dev_neg_rows]);
train_rows = sort([train_pos_rows, train_neg_rows]);
dev = x_train(dev_rows,:);
dev_labels = y_train(dev_rows,:);
x_train = x_train(train_rows,:);
y_train = y_train(train_rows,:);
s=sign(y_train);
ipositif=sum(s(:)==1);
inegatif=sum(s(:)==-1);
% disp(ipositif);
% disp(inegatif);

%x_test = [x_test power(x_test,2) power(x_test,3)];% x_test .* power(x_test,2)];

Xtrain_norm = x_train;
Xtest_norm = x_test;

[~,numCol] = size(Xtrain_norm);
[w,w_0] = train_ls(Xtrain_norm,y_train,1);
%w = inv(transpose(Xtrain_norm) * Xtrain_norm) * transpose(Xtrain_norm) * y_train;
%w_0 = 0;
col = ones(length(dev),1);
X_temp = [dev col];
w(numCol + 1) = w_0; 
predictY = X_temp*w;

[X,Y,T,AUC] = perfcurve(dev_labels,predictY,1);
disp(AUC);

%Xtrain_norm = x_train_copy;
%[~,numCol] = size(Xtrain_norm);
%[w,w_0] = train_ls(Xtrain_norm,y_train_copy,1);
col = ones(numTestSamples,1);
X_test = [Xtest_norm col];
%w(numCol + 1) = w_0; 
predictTest = X_test*w;

fileID = fopen('result.txt','w');
fprintf(fileID,'Id,Prediction\n');
fprintf(fileID, '%d,%1.4f\n', [transpose(x_test_id);transpose(predictTest)]);
fclose(fileID);