function loss = Compute_loss(X_batch, Ys_batch, ConvNet)
%Ys is one-hot encoded

[P, ~, ~] = EvaluateClassifier(X_batch, ConvNet);
sum_loss = sum(-log(dot(double(Ys_batch),P)));
N = size(X_batch,2); % Assuming the second column is N
loss = 1/N * sum_loss;

end
