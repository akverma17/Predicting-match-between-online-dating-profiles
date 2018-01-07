% load data
A = importdata('train.txt');
B = importdata('test.txt');
[numTrainSamples, numTrainDim] = size(A);
[numTestSamples, numTestDim] = size(B);
x_test_id = B(:,numTestDim);
y_train = A(:,numTrainDim);
[A, B] = normalizeAll(A, B);

% x_train = A(:,1:numTrainDim-1);
% x_test = B(:,1:numTestDim-1);


%x_train1 = A(:,1:58);
%x_train2 = A(:,59:numTrainDim-1);
x_train1 = [A(:,1:25) A(:,52:58)];
x_train2 = [A(:,59:83) A(:,110:numTrainDim-1)];
x_train1 = [x_train1(:,1:8) x_train1(:,10:12) x_train1(:,15:32)];
x_train2 = [x_train2(:,1:8) x_train2(:,10:12) x_train2(:,15:32)];
% x_train3 = zeros(numTrainSamples,1);
% x_train4 = zeros(numTestSamples,size(x_train1,2));
% for i = 1:numTrainSamples
%     x_train4(i,:) = (x_train1(i,:) .* x_train2(i,:)) / (norm(x_train1(i,:)) * norm(x_train2(i,:)));
%     for j = 1:size(x_train1,2)
%         x_train3(i,1) = x_train3(i,1) + x_train1(i,j) + x_train2(i,j);
%     end
% end
% x_train3 = zeros(numTrainSamples,1711);
% x_train4 = [x_train1 x_train2];
% x_train = x_train4;
% k = 1;
% for i = 1:size(x_train4,2)
%     for j = i+1:size(x_train4,2)
%         x_train3(:,k) = x_train4(:,i) .* x_train4(:,j);
%         k = k+1;
%     end
% end
x_train = [x_train1 x_train2];
%x_train = [x_train power(x_train,2)];% (x_train1 .* x_train2)];
%x_train = [x_train1 x_train2 x_train3];

%x_test1 = B(:,1:58);
%x_test2 = B(:,59:numTestDim-1);
x_test1 = [B(:,1:25) B(:,52:58)];
x_test2 = [B(:,59:83) B(:,110:numTrainDim-1)];
x_test1 = [x_test1(:,1:8) x_test1(:,10:12) x_test1(:,15:32)];
x_test2 = [x_test2(:,1:8) x_test2(:,10:12) x_test2(:,15:32)];
% x_test3 = zeros(numTestSamples,1);
% x_test4 = zeros(numTestSamples,size(x_test1,2));
% 
% for i = 1:numTestSamples
%     x_test4(i,:) = (x_test1(i,:) .* x_test2(i,:)) / (norm(x_test1(i,:)) * norm(x_test2(i,:)));
%     for j = 1:size(x_test1,2)
%         x_test3(i,1) = x_test3(i,1) + x_test1(i,j) + x_test2(i,j);
%     end
% end

% x_test3 = zeros(numTestSamples,1711);
% x_test4 = [x_test1 x_test2];
% x_test = x_test4;
% k = 1;
% for i = 1:size(x_test4,2)
%     for j = i+1:size(x_test4,2)
%         x_test3(:,k) = x_test4(:,i) .* x_test4(:,j);
%         k = k+1;
%     end
% end
x_test = [x_test1 x_test2];
%x_test = [x_test power(x_test,2)];% (x_test1 .* x_test2)];
%x_test = [x_test1 x_test2 x_test3] ;
% % first normalize all the data so that it lies between -1 and 1.
% %[Xtrain_norm, Xtest_norm] = normalizeAll(x_train, x_test);
% %[ytrain, ytest] = normalize(ytrain, ytest);
xPos = zeros(150000,size(x_train,2));
xNeg = zeros(150000,size(x_train,2));
yPos = ones(150000,1);
yNeg = ones(150000,1);
i = 1;
k = 1;
for j = 1:numTrainSamples
    if y_train(j,1) == 1
        if i <= 150000
            xPos(i,:) = x_train(j,:);
            yPos(i,1) = 1;
            i = i+1;
        end
    else
        if k <= 150000
            xNeg(k,:) = x_train(j,:);
            yNeg(k,1) = -1;
            k = k+1;
        end
    end
end

Xtrain_norm = [xPos;xNeg];
y_train_new = [yPos;yNeg];
%disp(y_train_new);
%Xtrain_norm = x_train;
Xtest_norm = x_test;
%[Xtrain_norm, Xtest_norm] = normalizeAll(x_train, x_test);

%Xtrain_norm = generate_poly_features(Xtrain_norm,2);
%Xtest_norm = generate_poly_features(Xtest_norm,2);

dev = Xtrain_norm(200001:300000,:);
Xtrain_norm = Xtrain_norm(1:200000,:);
dev_labels = y_train(200001:300000,:);
y_train = y_train(1:200000,:);
[~,numCol] = size(Xtrain_norm);
[w,w_0] = train_ls(Xtrain_norm,y_train,0);
col = ones(100000,1);
X_temp = [dev col];
w(numCol + 1) = w_0; 
predictY = X_temp*w;
% w = inv(transpose(Xtrain_norm) * Xtrain_norm) * transpose(Xtrain_norm) * y_train;
% predictY = dev * w;
[X,Y,T,AUC] = perfcurve(dev_labels,predictY,1);
disp(AUC);

col = ones(numTestSamples,1);
X_test = [Xtest_norm col];
predictTest = X_test*w;
%predictTest = Xtest_norm*w;

fileID = fopen('result.txt','w');
fprintf(fileID,'Id,Prediction\n');
fprintf(fileID, '%d,%1.4f\n', [transpose(x_test_id);transpose(predictTest)]);
fclose(fileID);