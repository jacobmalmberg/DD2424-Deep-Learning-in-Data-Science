function [P_batch, X_batch1, X_batch2] = EvaluateClassifier(X_batch,ConvNet)
%aka forward pass, p7
nlen=19; %hardcode?
[~,k1,~] = size(ConvNet.F{1});

MF1=MakeMFMatrix(ConvNet.F{1}, nlen);
X_batch1 = max(MF1*X_batch,0);

%nlen1=size(X_batch1,2)% p2 
nlen1 = nlen-k1+1;
MF2=MakeMFMatrix(ConvNet.F{2}, nlen1);
X_batch2 = max(MF2*X_batch1,0);

S_batch = ConvNet.W*X_batch2;
P_batch = softmax(S_batch); % ???

end