function [loss, a, h, o, p] = ComputeLoss(X, Y , RNN, h0)
%fwd pass
%assuming n is seq_length

K = size(RNN.c,1);
m = size(RNN.b,1);
n = size(X,2); 

ht = h0;


a=zeros(m,n);
h=zeros(m,n);
o=zeros(K,n);
p=zeros(K,n);

for t=1:n
    a(:,t) = RNN.W*ht + RNN.U*X(:,t) + RNN.b; %mx1
    h(:,t) = tanh(a(:,t)); %mx1
    ht = h(:,t); %needed for next iteration
    o(:,t) = RNN.V * h(:,t) +RNN.c; % Kx1
    p(:,t) = softmax(o(:,t));
end

h= [h0 h]; %insert h0 at the first place

loss = sum(-log(dot(double(Y),p)));
end

