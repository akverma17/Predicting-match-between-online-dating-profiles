function [w,w_0] = train_rr(X,y,lambda)
% get dimensions of matrix X
[row, col] = size(X);
% form a column vector of 1's and 0's
one = ones(row,1);
zero = zeros(col,1);
% making X_new = [transpose(X);transpose(1)] as given in the question means
% representing it column wise
X_new = [transpose(X); transpose(one)];
% defining I_new as given in question in column form
I_new = [eye([col col]), zero; transpose(zero), 0];
% defining C as given in question
C = X_new * transpose(X_new) + lambda * I_new;
% defining d as given in question
d = X_new * y;
% calculating w by the formula.
w = inv(C) * d;
% last row of w is the bias w_0
w_0 = w(end);
% deleting last row and the rest is w
w(end) = [];
end