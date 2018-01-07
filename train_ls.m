function [w,w_0] = train_ls(X,Y,bias)
% calculating the no. of samples.
[numSamples, ~] = size(X);
% if bias is one append a column of 1's to the X.
if bias == 1
    col = ones(numSamples,1);
    X = [X col];
end
% these are the steps if transpose(X)*X is not invertible.
[V,D,W] = eig(transpose(X)*X);
d = diag(D);
for i = 1:length(d)
    if d(i) ~= 0
        d(i) = 1/d(i);
    end
end
DPlus = diag(d);
% at last calculate the vector w.
w = V * DPlus * transpose(V) * transpose(X) * Y;
% if bias is 0, w_0 is 0 else it is the last value of vector w. Delete this
% last value from w and w is the remaining vector.
if bias == 0
    w_0 = 0;
else
    w_0 = w(end);
    w(end) = [];
end