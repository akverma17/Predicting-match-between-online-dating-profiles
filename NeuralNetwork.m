A = importdata('train.txt');
B = importdata('test.txt');
[numTrainSamples, numTrainDim] = size(A);
[numTestSamples, numTestDim] = size(B);
x_test_id = B(:,numTestDim);
y_train = A(:,numTrainDim);
x_train = A(:,1:numTrainDim-1);
x_test = B(:,1:numTestDim-1);
[x_train, x_test] = normalizeAll(x_train, x_test);

neg_rows = find(y_train == -1);
y_train(neg_rows,1) = 0;

dev = x_train(300001:numTrainSamples,:);
x_train = x_train(1:300000,:);
dev_labels = y_train(300001:numTrainSamples,:);
y_train = y_train(1:300000,:);

epsilon = 0.01; % learning rate for gradient descent
reg_lambda = 0.01; % regularization strength
disp('ready');
[W1,b1,W2,b2] = build_model(x_train, y_train, dev, dev_labels, 50, 10000, reg_lambda, epsilon);
yPredict = predict(W1,b1,W2,b2, x_test);
fileID = fopen('result.txt','w');
fprintf(fileID,'Id,Prediction\n');
fprintf(fileID, '%d,%1.4f\n', [transpose(x_test_id);transpose(predictTest)]);
fclose(fileID);

function[data_loss] = calculate_loss(X,y,W1,b1,W2,b2, reg_lambda)
%     disp(W1);
    z1 = X * W1 + b1;
    a1 = max(0,z1);
    z2 = a1 * W2 + b2;
    data_loss = z2 - log(sum(exp(z2)));
    data_loss = -1.0 * data_loss .* y;
    data_loss = sum(data_loss) / size(X,1);
%     exp_scores = exp(z2);
%     probs = exp_scores / sum(exp_scores);
%     corect_logprobs = -log(probs) .* y;
%     data_loss = sum(corect_logprobs) / size(X,1);
    data_loss = data_loss + reg_lambda/2 * (norm(W1) + norm(W2));
end

function[yPredict] =  predict(W1,b1,W2,b2, X)
    z1 = X * W1 + b1;
    a1 = max(0,z1);
    z2 = a1 * W2 + b2;
    exp_scores = exp(z2);
    probs = exp_scores / sum(exp_scores);
    yPredict = max(probs,[],2);
end

function[W1,b1,W2,b2] =  build_model(Xtr, ytr, Xdev, ydev, nn_hdim, num_passes, reg_lambda, epsilon)
    W1 = randn(size(Xtr,2), nn_hdim) / sqrt(size(Xtr,2));
    b1 = zeros(1, nn_hdim);
    W2 = randn(nn_hdim, 1) / sqrt(nn_hdim);
    b2 = zeros(1, 1);
    
    for i = 1:num_passes
        z1 = Xtr * W1 + b1;
%        disp(z1(1,:));
        a1 = max(0,z1);
  %      disp(a1(1,:));
        z2 = a1 * W2 + b2;
        exp_scores = exp(z2);
        probs = exp_scores / sum(exp_scores);

        %Backpropagation
        delta3 = (probs - ytr) / size(probs,1);
        delta2 = delta3 * transpose(W2); 
        delta2(a1 <= 0) = 0;
        dW2 = transpose(a1) * delta3;
        db2 = sum(delta3);
        dW1 = transpose(Xtr) * delta2;
        db1 = sum(delta2);
        
%         dW2 = transpose(a1) * delta3;
%         db2 = sum(delta3);
%         delta2 = delta3 * transpose(W2) .* (1 - power(a1, 2));
%         dW1 = transpose(Xtr) * delta2;
%         db1 = sum(delta2);

        % Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 = dW2 + reg_lambda * W2;
        dW1 = dW1 + reg_lambda * W1;

        % Gradient descent parameter update
        W1 = W1 - epsilon * dW1;
        b1 = b1 - epsilon * db1;
        W2 = W2 - epsilon * dW2;
        b2 = b2 - epsilon * db2;
  
        %if mod(i,1000) == 0
           loss = calculate_loss(Xtr,ytr,W1,b1,W2,b2, reg_lambda);
           yPredict = predict(W1,b1,W2,b2,Xdev);
           [~,~,~,AUC] = perfcurve(ydev,yPredict,1);
           disp('iteration :');
           disp(i);
           disp('loss :');
           disp(loss);
           disp('AUC :');
           disp(AUC);
        %end
    end
end