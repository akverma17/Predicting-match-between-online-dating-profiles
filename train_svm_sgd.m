function [w] = train_svm_sgd(X,y,C,a,T)
    
    % calculate no. of samples and dimensions
    [numSamples, numDim] = size(X);
    
    % w is (10000,numDim) matrix to store w values at each iteration. Start
    % with value of 1 in all dimensions.
    
    w = ones(T,numDim);
    
    % error matrix to store hinge loss at each iteration.
    
    error = zeros(T,1);
    
    % obj matrix to store objective value at each iteration.
    
    obj = zeros(T,1);
    
    % numIter matrix to store the iterations. First iteration will be 1.
    
    numIter = zeros(T,1);
    numIter(1,1) = 1;
    
    % starting objective value in which w = 1 and loss is maximum
    
    obj(1,1) = 0.5 * (norm(w(1,:))^2) + C * numSamples;
    
    % iterate from 2nd iteration onwards
    for i = 2:T
        
    % learning rate = a/i
        eta = a/i;
        
        sum = 0.0;
        
    % store the iterations
        numIter(i,1) = i;
        
    % iterate through all the samples
        for j = 1:numSamples
            
    % calculate the hinge loss and sum over all the samples
            s = max(0,1-(y(j,1) * dot(w(i-1,:),X(j,:))));
            error(i,1) = error(i,1) + s;
            
    % if hinge loss is >= 0.001, gradient is -yx else 0. sum the gradients
    % over all the training samples
            if s >= 0.001
                sum = sum + y(j,1) * X(j,:);
            end
        end
        
    % update rule for w. same as derived in 1st part
        w(i,:) = (1-eta)*w(i-1,:) + eta*C*sum;
        
    % also store the objective value at each iteration to avoid same
    % computation again.
        obj(i,1) = 0.5 * (norm(w(i,:))^2) + C * error(i,:);
    end
    
    % display the final objective value.
    
    disp('objective : ');
    disp(obj((size(obj,1)),1));
    
    % plot the log-log plot of objective value at each iteration.
    
    figure(1)
    p1 = scatter(log(numIter(:,1)),log(obj(:,1)),10,'+','MarkerEdgeColor',[0.5 0 0],...
              'MarkerFaceColor',[0.7 0 0],...
              'LineWidth',1.5);