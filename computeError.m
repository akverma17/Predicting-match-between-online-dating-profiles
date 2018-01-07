function err = computeError(preds, Y)
	err = sum(preds ~= Y) / length(Y) * 100;
end