function [w,w_0] = train_svm_primal(X,y,C)

[row, col] = size(X);
% H is the diagonal matrix of size row+col+1 with col*col identity matrix
% as its submatrix for each dimension of w and rest all 0 for bias and
% slack variables.

H = diag( [ones(1,col) , zeros(1,row + 1)]);

% f is a column vector with 0's for w_i and bias, C for slack variables.

f = [zeros(1,col),0,C * ones(1,row)];

% taking transpose of f to multiply with x

f = transpose(f);
A = X;

% multiplying -y with X to get -y_i*X_i
for i = 1:row
    A(i,:) = A(i,:)* -y(i,1);
end

% concatenating -yX, -y and identity matrix of size m so that on
% multiplication  with x gives -eta - y(<w,x>+b) <= -1

A = [A -1*y -1 * eye(row)];

% b is column vector of -1's

b = -1 * ones(row,1);

% lower bound for slack variables is 0, for rest it is -inf

lb = [-inf(col+1,1) ; zeros(row,1)];

% upper bound for all parameters is +inf

ub = inf(row+col+1,1);

% Aeq and Beq are null

[x,obj] = quadprog(H,f,A,b,[],[],lb,ub);
disp(obj);

% calculating objective value and no. of support vectors

objective = 0.5 * transpose(x) * H * x + transpose(f)*x;

disp('objective:');
disp(objective);
numSupportVectors = 0;
w = x(1:col+1);
w_0 = w(end);
w(end) = [];
epsi = x(col + 2 : end);
for i = 1 : row
    value =  X(i , :) * w + w_0;
    diff = abs(1 - value);
    if epsi(i) > 0.001 || diff < 0.001
        numSupportVectors = numSupportVectors + 1;
    end
end
disp('num: ')
disp(numSupportVectors);
end
