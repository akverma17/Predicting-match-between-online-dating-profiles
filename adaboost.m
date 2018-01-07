A = importdata('train.txt');
B = importdata('test.txt');
[numTrainSamples, numTrainDim] = size(A);
[numTestSamples, numTestDim] = size(B);
x_test_id = B(:,numTestDim);
y_train = A(:,numTrainDim);
x_train = A(:,1:numTrainDim-1);
x_test = B(:,1:numTestDim-1);
x_train = zscore(x_train);
x_test = zscore(x_test);

a = x_train(:,1:58);
b = x_train(:,59:size(x_train,2));
c = x_test(:,1:58);
d = x_test(:,59:size(x_test,2));
e = a .* b;
f = c .* d;
%x_train = [x_train power(x_train,2) e];% power(x_train,3) power(e,2) (a .* e) (b .* e) power(x_train,4) e .* power(a,2) e .* power(b,2) power(x_train,5) e .* power(a,3) a.* power(e,2) b .* power(e,2) e .* power(b,3)];
%x_test = [x_test power(x_test,2) f];% power(x_test,3) power(f,2) (c .* f) (d .* f) power(x_test,4) f .* power(c,2) f .* power(d,2) power(x_test,5) f .* power(c,3) c .* power(f,2) d .* power(f,2) f .* power(d,3)];


dev = x_train(300001:numTrainSamples,:);
x_train = x_train(1:300000,:);
dev_labels = y_train(300001:numTrainSamples,:);
y_train = y_train(1:300000,:);

pos_row = find(y_train == 1);
neg_row = find(y_train == -1);

D = ones(1,size(x_train,1));
D(1,pos_row) = 1 ./ length(pos_row);
D(1,neg_row) = 1 ./ length(neg_row);
T = 10;
w = zeros(1,T);
th = zeros(1,T);
dim = zeros(1,T);
for i = 1:T
    [d,theta] = weak_learner(D,x_train,y_train);
    disp(d);
    disp(theta);
    h = repmat(theta,size(x_train,1),1) - x_train(:,d);
    epsilon = sign(h) ~= y_train;
%     disp(epsilon);
    epsilon = dot(D,transpose(epsilon));
    w(1,i) = 0.5 * log(1/epsilon - 1);
    disp(w(1,i));
    th(1,i) = theta;
    dim(1,i) = d;
%     D = D .* transpose(exp(repmat(-w(1,i),size(x_train,1),1) .* (y_train .* sign(h))));
    sum = 0;
    for j = 1:size(D,2)
        D(1,j) = D(1,j) * exp( -1 * w(1,i) * y_train(j,1) * sign(h(j,1)));
        sum = sum + D(1,j);
    end
    D = D ./ repmat(sum,1,size(D,2));
end
% disp([transpose(w) transpose(th) transpose(dim)]);
predictY = zeros(length(dev),1);
for i = 1:T
    h = sign(repmat(th(1,i),length(dev),1)  - dev(:,dim(1,i)));
    predictY = predictY + repmat(w(1,i),length(dev),1) .*  h;
end

% predictY = sign(predictY);
[X,Y,~,AUC] = perfcurve(dev_labels,predictY,1);
disp(AUC);

predictTest = zeros(numTestSamples,1);

for i = 1:T
    h = sign(repmat(th(1,i),numTestSamples,1)  - x_test(:,dim(1,i)));
    predictTest = predictTest + repmat(w(1,i),numTestSamples,1) .*  h;
end
% disp(predictTest);
% predictTest = sign(predictTest);

fileID = fopen('result.txt','w');
fprintf(fileID,'Id,Prediction\n');
fprintf(fileID, '%d,%1.4f\n', [transpose(x_test_id);transpose(predictTest)]);
fclose(fileID);

function [d,theta] = weak_learner(D,x_train,y_train)
    [numSamples,numDim] = size(x_train);
    F = intmax;
    for j = 1:numDim
        X = [x_train y_train];
        X = sortrows(X,j);
        y_train = X(:,size(X,2));
        X = X(:,1:size(X,2)-1);
        rows = find(y_train == 1);
        f = sum(D(1,rows));
        if f < F
            F = f;
            theta = X(1,j)-1;
            d = j;
        end
        for i = 1:numSamples-1
            f = f - y_train(i,1) * D(1,i);
            if (f < F) && (X(i,j) ~= X(i+1,j))
                F = f;
                theta = 0.5 * (X(i,j) + X(i+1,j));
                d = j;
            end
        end
    end
end

