%b?rja vid den r?da gubben
load('inData.mat') %load book_data, char to ind etc

m = 100;
seq_length = 25;
K = length(char_to_ind); % 79

sig = 0.01;

rng(400); % 400 for life
%rng('shuffle')

% RNN.b = zeros(m,1);
% RNN.c = zeros(K,1);
% RNN.U = randn(m, K)*sig;
% RNN.W = randn(m, m)*sig;
% RNN.V = randn(K, m)*sig;
% 

%s?tt till 1 om bara vill k?ra 1 n?tverk
number_of_networks = 4;
RNN_cell = cell(number_of_networks,1);


RNN_cell{1} = CreateNetwork(400, 400, K, sig, 0.02);
RNN_cell{2} = CreateNetwork(400, 400, K, sig, 0.01);
RNN_cell{3} = CreateNetwork(400, 600, K, sig, 0.05);
RNN_cell{4} = CreateNetwork(400, 600, K, sig, 0.02);

result_cell = cell(number_of_networks,3); %first is loss, second is RNN, third is hprev

%gradient checks, remember to set m=5

% h0 = zeros(m,1);
% x0 = zeros(K,1);
% %n=5;
% 
% %Y = synt_text(RNN, h0,x0,n);
% 
% X_chars = book_data(1:seq_length);
% Y_chars = book_data(2:seq_length+1);
% 
% X_chars_one_hot = one_hot_encode_string(X_chars, char_to_ind);
% Y_chars_one_hot = one_hot_encode_string(Y_chars, char_to_ind);
% h=1e-4;
% num_grads = ComputeGradsNum(X_chars_one_hot, Y_chars_one_hot, RNN, h);
% 
% [loss, a, h, o, p] = ComputeLoss(X_chars_one_hot, Y_chars_one_hot, RNN, h0);
% grads = ComputeGradients(X_chars_one_hot, Y_chars_one_hot, RNN, a, h, p);
% 
% grad_b_rel_error = abs(grads.b - num_grads.b) ./ max(eps, abs(grads.b) + abs(num_grads.b));
% grad_c_rel_error = abs(grads.c - num_grads.c) ./ max(eps, abs(grads.c) + abs(num_grads.c));
% grad_U_rel_error = abs(grads.U - num_grads.U) ./ max(eps, abs(grads.U) + abs(num_grads.U));
% grad_W_rel_error = abs(grads.W - num_grads.W) ./ max(eps, abs(grads.W) + abs(num_grads.W));
% grad_V_rel_error = abs(grads.V - num_grads.V) ./ max(eps, abs(grads.V) + abs(num_grads.V));
% 
% max(grad_b_rel_error)
% max(grad_c_rel_error)
% max(max(grad_U_rel_error))
% max(max(grad_W_rel_error))
% max(max(grad_V_rel_error))


dataset_length = size(book_data,2);

n_epochs = 10;
e_values = 1:seq_length:dataset_length-seq_length;
update_steps = length(e_values)*n_epochs;

smooth_loss_vector = zeros(update_steps,1);
smooth_loss = 0;

chars_to_synt = 200;

text_cell = cell(idivide(int32(update_steps),10000)+1,1);
save_iter=1;

for k=1:length(RNN_cell)
    %to test multiple RNNs
    k
    RNN =  RNN_cell{k};
    
    %init adagrad.
    adagrad.b = zeros(size(RNN.b));
    adagrad.c = zeros(size(RNN.c));
    adagrad.U = zeros(size(RNN.U));
    adagrad.W = zeros(size(RNN.W));
    adagrad.V = zeros(size(RNN.V));

    iter=1;
    save_iter=1;
    
    %graph stuff
    smooth_loss_vector = zeros(update_steps,1);
    smooth_loss = 0;
    for i=1:n_epochs
        %e=1;
        i
        m = size(RNN.W,1);
        hprev = zeros(m,1); 
        h0 = hprev;

        for e=e_values
            X_chars = book_data(e:e+seq_length-1);
            Y_chars = book_data(e+1:e+seq_length);

            X_chars_one_hot = X(:,e:e+seq_length-1);
            Y_chars_one_hot = X(:,e+1:e+seq_length);


            [RNN, adagrad, loss, hprev] = MiniBatchGD(X_chars_one_hot, Y_chars_one_hot, RNN, adagrad, hprev);
            loss;
            if smooth_loss == 0
                smooth_loss = loss;
            end

            smooth_loss = 0.999*smooth_loss + 0.001*loss;
            smooth_loss_vector(iter) = smooth_loss;

%             if (iter == 1) || (mod(iter,10000) == 0 && iter <=100000) %|| (e == dataset_length-seq_length)
%                 e;
%                 smooth_loss;
%                 iter
%                 Y = synt_text(RNN, h0, X_chars_one_hot(:,1), 200);
%                 text_cell{save_iter} = DecodeString(Y, ind_to_char)
%                 save_iter = save_iter + 1;
% 
%             end

            h0=hprev;
            iter= iter+1;


        end
    end
    result_cell{k,1} = smooth_loss_vector;
    result_cell{k,2} = RNN;
    result_cell{k,3} = hprev;
end

function [RNN, adagrad, loss, hprev] = MiniBatchGD(X_chars_one_hot, Y_chars_one_hot, RNN, adagrad, hprev)
[loss, a, h, o, p] = ComputeLoss(X_chars_one_hot, Y_chars_one_hot, RNN, hprev);
grads = ComputeGradients(X_chars_one_hot, Y_chars_one_hot, RNN, a, h, p);
hprev = h(:,end);


%sgd
 
eps = 1e-9;
for f = fieldnames(grads)'
    %clipping
    grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    
    %adagrad
    adagrad.(f{1}) = adagrad.(f{1}) + grads.(f{1}).^2;
    RNN.(f{1}) = RNN.(f{1}) - (RNN.eta*grads.(f{1}))./((adagrad.(f{1})+eps).^(0.5));
end
end
        



function Y = synt_text(RNN, h0, x0, n)
%0.3
%assuming n is like a timestep

K  =size(RNN.C,1);
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



function xnext = sample_char(p, K)
cp = cumsum(p);
a = rand;
ixs = find(cp -a >0);
ii = ixs(1);
xnext = one_hot_encode_char(ii, K);

end




function [loss, a, h, o, p] = ComputeLoss(X, Y , RNN, h0)
%h(1) is hprev
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



function grads = ComputeGradients(X, Y, RNN, a, h, p)
%hprev should be h(1)

m = size(RNN.U,1);
n = size(X,2);

grads.b = zeros(size(RNN.b)); %mx1
grads.c = zeros(size(RNN.c)); %kx1
grads.U = zeros(size(RNN.U)); %mxK
grads.W = zeros(size(RNN.W)); %mxm
grads.V = zeros(size(RNN.V)); %Kxm

g_batch = -(Y-p)'; %NxK, this is dL/do_t

grads.c = g_batch'*ones(n,1); %Kx1

%h is mxn
%only use h_t != 0. therefore start at 2. h(1) is h0.
grads.V = g_batch'*h(:,2:end)'; % KxN * NxM = KxM


%now grads for a & h
dL_dA = zeros(n,m); % one per time step?

dL_dh_tao = g_batch(end,:)*RNN.V; %1xK * KxM = 1xM

dL_dA_tao = dL_dh_tao*diag(1-tanh(a(:,end)).^2); %1xM * MxM = 1xM

dL_dA(end,:) = dL_dA_tao;

for i=n-1:-1:1
    dL_dh_t = g_batch(i,:)* RNN.V + dL_dA(i+1,:)*RNN.W; % 1xM + MxM = 1xM
    dL_dA(i,:) = dL_dh_t*diag(1-tanh(a(:,i)).^2); % 1xM *MxM = 1xM
    
end

%gradW

%dL_dA_t * h_t-1. this will work because h(1) = h0. never use h_tao?
% for t=1:n
%     grads.W = grads.W + dL_dA(t,:)'*h(:,t)'; % Mx1 * 1xM = MxM
% end

grads.W = dL_dA'*h(:,1:end-1)';

grads.U = dL_dA'*X';
grads.b = dL_dA'*ones(n,1); % Mx1

end



function one_hot_char = one_hot_encode_char(index, K)
one_hot_char = zeros(K,1);
one_hot_char(index)=1;

end



function one_hot_matrix = one_hot_encode_string(char_string, char_to_ind)
%K = 79;% hardcode 4 life
N = length(char_string);
one_hot_matrix = zeros(K,N); 
for i=1:N
    index = char_to_ind(char_string(i));
    one_hot_matrix(index,i)=1;
end
end

function string = DecodeString(Y, ind_to_char)
string=[];
[~,I] = max(Y, [], 1); %argmax

for i=1:length(I)
    string = [string ind_to_char(I(i))];
end
end

function RNN = CreateNetwork(seed, m, K, sig, eta)
rng(seed); % 400 for life
RNN.eta = eta;
RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;

end


