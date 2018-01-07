function [X_train_norm, X_test_norm] = normalizeAll(X_train, X_test)
[row , col] = size(X_train);
X_train_norm = X_train;
X_test_norm = X_test;
% calculate min value and max value row in X
minVal = min(X_train);
maxVal = max(X_train);

for i = 1 : row 
    trainRow = X_train(i,:);
    % subtract the min value row from each row and divide by
    % (maxVal-minVal) to normalize it between 0 and 1.
    newRow = trainRow - minVal;
    newRow = newRow ./ (maxVal-minVal); 
    %now scale the values to lie in range -1 to 1 instead of 0 to 1
    newRow = newRow * 2 - 1;
    X_train_norm(i,:) = newRow;
end

[row , col] = size(X_test);
minVal = min(X_test);
maxVal = max(X_test);
for i = 1 : row 
    testRow = X_test(i,:);
    % subtract the min value row from each row and divide by
    % (maxVal-minVal) to normalize it between 0 and 1.
    newRow = testRow - minVal;
    newRow = newRow ./ maxVal; 
    %now scale the values to lie in range -1 to 1 instead of 0 to 1
    newRow = newRow * 2 - 1;
    X_test_norm(i,:) = newRow;
end

end