function J = ComputeCost(X, Y, W, b, lambda)
%4
%Y is one-hot encoded
W_sumsqr=sumsqr(W); % https://se.mathworks.com/help/deeplearning/ref/sumsqr.html
P = EvaluateClassifier(X, W, b);
sum_loss = sum(-log(dot(double(Y),P)));
N = size(X,2); % Assuming the second column is N
J = 1/N * sum_loss + lambda*W_sumsqr; 
end
