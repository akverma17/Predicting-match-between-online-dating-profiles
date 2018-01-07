function [classifiers, classifiersWeights] = adaBoostTrain(X, Y, nbIter)
	[n, p] = size(X);
	dataWeights = ones(n, 1) / n;
    pos = find(Y == 1);
    neg = find(Y == -1);
    dataWeights(pos,1) = 1 ./ length(pos);
    dataWeights(neg,1) = 1 ./ length(neg);

	classifiers = [];
	classifiersWeights = [];
	for i = 1:nbIter
		classifier = decisionStumpTrain(X, Y, dataWeights);

		% Get predicted labels
		labelsPred = decisionStumpVal(classifier, X);
		errorWeighted = sum((Y ~= labelsPred)' * dataWeights);

		% Compute the weight of the current classifier
		classifierWeight = 1/2 * log((1 - errorWeighted) / errorWeighted);

		% Recompute data weights
		for j = 1:n
			dataWeights(j) = dataWeights(j) * exp(-classifierWeight * Y(j) * labelsPred(j));
		end

		% Renormalize data weights
		totalWeights = sum(dataWeights);
		dataWeights = dataWeights / totalWeights;

		% Remember the current classifier and its weight
		classifiers = [classifiers; classifier];
		classifiersWeights = [classifiersWeights; classifierWeight];
        disp(i);
	end
end