function [X_poly] = generate_poly_features(X,k)
X_poly = [];
for j = 1 : k
    % for each value j from 1 to k, calculate power of the matrix times j.
    trainRow = power(X,j);
    % on each iteration size of each matrix increases. Thus to append the new
    % matrix generated, make a grid of size equal to the combined size of both
    % the matrices and then accumulate the values of both the matrices in that
    % new big matrix.
    [i1,j1] = ndgrid(1:size(X_poly,1),1:size(X_poly,2));
    [i2,j2] = ndgrid(1:size(trainRow,1),(1:size(trainRow,2))+size(X_poly,2));
    X_poly = accumarray([i1(:),j1(:);i2(:),j2(:)],[X_poly(:);trainRow(:)]);
end