function [w,w_0] = train_perceptron(X,y)
[row, col] = size(X);
X = [X ones(row,1)];
% initialize w = 0 in the beginning
w = zeros(1,col+1);
% iterate through every point and if y<w,x> < 0, update w.
for i = 1:row
    if  sign(dot(w,X(i,:))) ~= sign(y(i,1))
        w = w + y(i,1)*X(i,:);
    end
end
w = transpose(w);
% last row of w is the bias w_0
w_0 = w(end);
% deleting last row and the rest is w
w(end) = [];
end