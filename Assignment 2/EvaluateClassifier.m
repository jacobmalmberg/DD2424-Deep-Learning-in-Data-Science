function [P, s1, s2, h] = EvaluateClassifier(X, W, b)
% pt3 in ass1, in ass2 this is exc2 pt1

s1 = W{1}*X +b{1};
h = max(0,s1);
s2 = W{2}*h + b{2};
P = exp(s2)./sum(exp(s2)); %dot for vectorwise output/division
%P = softmax(s2);
end