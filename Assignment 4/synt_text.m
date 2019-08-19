function Y = synt_text(RNN, h0, x0, n)
%0.3
%assuming n is like a timestep

K  =size(RNN.c,1);
Y = zeros(K,n);

h_t = h0;
xt = x0;



for t=1:n
    a_t = RNN.W*h_t + RNN.U*xt + RNN.b; %mx1
    h_t = tanh(a_t); %mx1
    o_t = RNN.V * h_t +RNN.c; % Kx1
    p_t = softmax(o_t);
    xnext = sample_char(p_t, K);
    Y(:,t) = xnext;
    xt = xnext;
end


end
