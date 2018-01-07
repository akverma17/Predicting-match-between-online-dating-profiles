function preds = adaBoostPredict(X, classifiers, classifiersWeights)

	classifiersPredictions = [];
	% Get predictions for each classifier
	for i = 1:length(classifiers)
		classifiersPrediction = decisionStumpVal(classifiers(i), X);
		classifiersPredictions = [classifiersPredictions classifiersPrediction];
	end
	% Multiply prediction for each classifier by its weight
	out = classifiersPredictions * classifiersWeights;
	preds = out;
end