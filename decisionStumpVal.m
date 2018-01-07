function EY = decisionStumpVal(classifier, X);
	nX = size(X,1);
	nC = size(X,2);

	if classifier.sens == '<'
	    idPos = find(X(:, classifier.icarac) > classifier.seuil);
	else
	    idPos = find(X(:, classifier.icarac) < classifier.seuil);
	end

	EY = -ones(nX, 1);
	EY(idPos) = 1;
end