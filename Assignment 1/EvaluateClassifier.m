function P = EvaluateClassifier(X, W, b)
%3
s = W*X +b;
P = exp(s)./sum(exp(s)); %dot for vectorwise output/division
end