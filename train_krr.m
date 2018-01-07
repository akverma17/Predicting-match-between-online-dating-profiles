function [alpha] = train_krr(X,y,lambda,kernel,gamma)

% calculate no. of samples as row and dimensions as col.
[row, col] = size(X);

% alpha will be calculated by eq derived from 5b. 
% X * transpose(X) is done by kernels sample by sample wise.
% resultant would be a matrix of size (row,row)
k = zeros(row,row);
if(strcmp(kernel,'linear') == 1)
    for i = 1 : row
        for j = 1 : row
            k(i,j) = linearK(X(i,:) , X(j,:));
        end
    end
else
    for i = 1 : row
        for j = 1 : row
            k(i,j) = gaussianK(X(i,:) , X(j,:), gamma);
        end
    end
end

alpha = inv(k + row*lambda*eye(row)) * y * row * lambda;

% function to calculate guassian kernel of two row vectors x1 and x2.
function [k] = gaussianK(x1,x2,gamma)
k = exp(-gamma*((norm(x1-x2))^2));

% function to calculate linear kernel of two row vectors x1 and x2.
function [k] = linearK(x1,x2)
k = dot(x1,x2);