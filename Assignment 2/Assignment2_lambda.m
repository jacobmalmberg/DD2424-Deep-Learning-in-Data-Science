%for coarse and fine search of lambda
addpath Datasets/cifar-10-batches-mat/

%[trainX, training_one_hot, trainY] = LoadBatch('data_batch_1.mat');
%[validationX, validation_one_hot, validationY] = LoadBatch('data_batch_2.mat');
[testX, test_one_hot, testY] = LoadBatch('test_batch.mat');


[aX, a_one_hot, aY] = LoadBatch('data_batch_1.mat');
[bX, b_one_hot, bY] = LoadBatch('data_batch_2.mat');
[cX, c_one_hot, cY] = LoadBatch('data_batch_3.mat');
[dX, d_one_hot, dY] = LoadBatch('data_batch_4.mat');
[eX, e_one_hot, eY] = LoadBatch('data_batch_5.mat');

validationX = aX(:,1:5000);
validation_one_hot = a_one_hot(:,1:5000);
validationY = aY(1:5000);

trainX = [aX(:,5001:10000) bX cX dX eX];
training_one_hot =[a_one_hot(:,5001:10000) b_one_hot c_one_hot d_one_hot e_one_hot];
trainY =[aY(5001:10000); bY; cY; dY; eY];



% set mean and center the data
mean_X = SetMeanX(trainX); % set the mean of the data
std_x = std(trainX, 0, 2);    
trainX = CenterDataset(trainX, mean_X, std_x); %center this too? top of p5
validationX = CenterDataset(validationX, mean_X, std_x);
testX = CenterDataset(testX, mean_X, std_x);

% sanity check #3
% trainX = trainX(:,1:100);
% training_one_hot = training_one_hot(:,1:100);
% 
% validationX = validationX(:,1:100);
% validation_one_hot = validation_one_hot(:,1:100);

d = 3072;
N = size(trainX,2);
K = 10;
m = 50;

% epoch etc settings
rng(400); % seed
n_batch = 100;
%eta = .01;
n_epochs = 8; % 4*45k = 180k %18 för bästa
lambda = 0.01;
%update_steps = (N*n_epochs)/n_batch;


% cyclical settings

eta_min = 1e-5;
eta_max = 1e-1;
n_s = 900;%2*floor(N/n_batch); % = 900; 
l=0;
cycles=2;
n_epochs = cycles*2*n_s*n_batch/N;
t= 2*l*n_s +1; %%to start at 1
t_end = 2*n_s*cycles; % last t value
eta_t_mat = get_cyclical_eta_array(t, l, cycles, eta_max, eta_min, n_s);
update_steps = cycles*9; %27 för n_s = 800. 9 för n_s = 500?
x_axis = round(linspace(0, t_end, update_steps)); % x-axeln. bör ha 9 steg per cykel

% lambda settings
l_min = 4.8793e-05;% -5;
l_max = 0.0047306;%-1;
l_count = 15; %generate 15 lambdas
lambda_array = generate_fine_lambda(l_min, l_max, l_count);
lambda_array = sort(lambda_array); % lowest to highest;
%lambda_array(1) = 0.0021622; %best

results = [];
for k=1:l_count
    [W, b] = InitWB(K, m ,d);
    best_acc = 0;
    
    k
    t = 1;
    results = [results; "lambda" lambda_array(k)];
    results = [results; "eta" "val_acc"];
    for i=1:n_epochs
        for j=1:N/n_batch
            eta = eta_t_mat(t);
            
            j_start = (j-1)*n_batch+1;
            j_end = j*n_batch;
            inds = j_start:j_end;
            X_batch = trainX(:, inds);

            %Y_batch = trainY(:, inds);
            training_one_hot_batch = training_one_hot(:, j_start:j_end);
            [W, b] = MiniBatchGD(X_batch, training_one_hot_batch, W, b, lambda_array(k), eta);

            if (t ==1 ) || (ismember(t,x_axis))  %(t == 1) || (mod(t,125) == 0)%(mod(t-2,130) == 0) || (t==1001) %(mod(t-2,190) == 0) || (t==4801)
                t;
                val_acc = ComputeAccuracy(validationX, validationY, W, b);
                if val_acc > best_acc
                    best_results=[eta val_acc];                   
                end
            end
            t = t+1;

        end


    end
    results = [results; best_results];
    results = [results; "-" "-"];
end
    
%accuracy, create plot, visualize W
acc = ComputeAccuracy(testX,testY,W,b);
% name_array = ['Ass 2. Cost. n_s= ',num2str(n_s),'. validation acc = ', num2str(validation_acc(end))];
% make_plot(x_axis, train_cost, validation_cost, 'update step', 'cost', name_array)
% 
% name_array = ['Ass 2. Loss. n_s= ',num2str(n_s),'. validation acc = ', num2str(validation_acc(end))];
% make_plot(x_axis, train_loss, validation_loss, 'update step', 'loss', name_array)
% 
% name_array = ['Ass2. Accuracy. n_s= ',num2str(n_s),'. validation acc = ', num2str(validation_acc(end))];
% make_plot(x_axis, train_acc, validation_acc, 'update step', 'accuracy', name_array)

function make_plot(x_axis, train_vec, validation_vec, x_label, y_label, name_array)
% to make plots like fig3/4
figure;
plot(x_axis, train_vec, x_axis, validation_vec);
legend('training','validation')
ylabel(y_label); 
xlabel(x_label);
path = "Result_Pics/";
name = name_array;
title(name)
saveas(gcf, path + name +".png") %gcf = current
end

function [X, Y, y] = LoadBatch(filename)
%pt 1
A = load(filename);
X = single(A.data');
%X = double(A.data')/255;   %convert values to 0-1  
y = double(A.labels +1); % adjust for matrix 1 indexing
%Y = bsxfun(@eq, y(:), 1:max(y)); % https://stackoverflow.com/questions/38947948/how-can-i-hot-one-encode-in-matlab
Y = y == 1:max(y);
Y= double(Y');
end

function [P, s1, s2, h] = EvaluateClassifier(X, W, b)
% pt3 in ass1, in ass2 this is exc2 pt1

s1 = W{1}*X +b{1};
h = max(0,s1);
s2 = W{2}*h + b{2};
P = exp(s2)./sum(exp(s2)); %dot for vectorwise output/division
end

function J = ComputeCost(X, Y, W, b, lambda)
%4 in ass1, now exc2 pt1
%Y is one-hot encoded
W_sumsqr=sumsqr(W); % https://se.mathworks.com/help/deeplearning/ref/sumsqr.html
[P, ~, ~, ~] = EvaluateClassifier(X, W, b);
sum_loss = sum(-log(dot(double(Y),P)));
N = size(X,2); % Assuming the second column is N
J = 1/N * sum_loss + lambda*W_sumsqr; 

end

function loss = ComputeLoss(X, Y, W, b, lambda)
%4 same as cost, but without regularization
%Y is one-hot encoded
[P, ~, ~, ~] = EvaluateClassifier(X, W, b);
sum_loss = sum(-log(dot(double(Y),P)));
N = size(X,2); % Assuming the second column is N
loss = 1/N * sum_loss;

end

function acc = ComputeAccuracy(X,y,W,b)
%5
[P, ~, ~, ~] = EvaluateClassifier(X,W,b);
[~,I] = max(P, [], 1); %argmax
N = size(y,1);
I=I';
acc = sum(y==I)/N;
end


function [grad_W_cell, grad_b_cell] = ComputeGradients(X, Y, W, b, lambda)
%6
%X, Y are minibatches (DxN)

N = size(Y,2);
[P, ~, ~, h] = EvaluateClassifier(X, W, b);

g_batch = -(Y-P); % KxN

grad_b2 = 1/N * (g_batch*ones(N,1));  %KxN * Nx1 = Kx1
grad_W2 = 1/N * (g_batch*h') + 2*lambda*W{2}; % KxN * NxM = KxM

g_batch = W{2}' * g_batch; % MxK * KxN = MxN;

g_batch = g_batch.*(h>0); % MxN

grad_W1 = 1/N * (g_batch*X') + 2*lambda*W{1};
grad_b1 = 1/N * (g_batch*ones(N,1));

% nu till det snurriga
% 
% d = diag(0< sum(s1,2)); %MxM. not square => sum
% 
% g_batch=g_batch*d; %NxM * MxM = NxM
% 
% grad_b1 = 1/N * sum(g_batch); % sum(NxM * MxM) = 1xM
% grad_b1 = grad_b1'; % Mx1
% 
% grad_W1 = 1/N * (g_batch'*X') +2*lambda*W{1}; % MxN * Nxd = Mxd;

grad_W_cell = cell(2,1);
grad_b_cell = cell(2,1);

grad_W_cell{1}= grad_W1;
grad_W_cell{2}= grad_W2;

grad_b_cell{1}= grad_b1;
grad_b_cell{2}= grad_b2;


end

function [Wstar, bstar] = MiniBatchGD(X, Y, W, b, lambda, eta)

[grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda);

Wstar = cell(2,1);
bstar = cell(2,1);

W1star = W{1} - eta*grad_W{1};
W2star = W{2} - eta*grad_W{2};
b1star = b{1} - eta*grad_b{1};
b2star = b{2} - eta*grad_b{2};

Wstar{1} = W1star;
Wstar{2} = W2star;

bstar{1} = b1star;
bstar{2} = b2star;
end

function mean_X = SetMeanX(X)
% set the mean for the dataset
mean_X = mean(X,2);
end

function dataset_vector = CenterDataset(X, mean_X, std_x)
% to set the mean of the dataset to zero
X = X - repmat(mean_X, [1, size(X, 2)]);
dataset_vector = X ./ repmat(std_x, [1, size(X, 2)]);
end

function eta_t = get_cyclical_eta(t, l, eta_max, eta_min, n_s)
% eq 14 and 15
if (2*l*n_s <= t) && (t <= (2*l +1)*n_s)
    eta_t = eta_min + ((t-2*l*n_s)/n_s)*(eta_max-eta_min);
elseif ((2*l+1)*n_s <= t) && (t <= 2*(l+1)*n_s)
    eta_t = eta_max - ((t-(2*l+1)*n_s)/n_s)*(eta_max-eta_min);
end
end

function eta_t_mat = get_cyclical_eta_array(t, l, cycles, eta_max, eta_min, n_s)

eta_t_mat = zeros(2*n_s*cycles,0);
steps_per_cycle = 2*n_s;
i=0;

while i < cycles
    
    if (t > 2*(l+1)*n_s)
        t= 2*l*n_s;
        i=i+1;
    end
        
    if (2*l*n_s <= t) && (t <= (2*l +1)*n_s)
        eta_t = eta_min + ((t-2*l*n_s)/n_s)*(eta_max-eta_min);
        eta_t_mat(i*steps_per_cycle+t) = eta_t;

    elseif ((2*l+1)*n_s <= t) && (t <= 2*(l+1)*n_s)
        eta_t = eta_max - ((t-(2*l+1)*n_s)/n_s)*(eta_max-eta_min);
        eta_t_mat(i*steps_per_cycle+t) = eta_t;
        
    end
    t = t+1;
    
end
end

function lambda_array = generate_lambda(l_min, l_max, l_count)
%to generate the lambdas
lambda_array=zeros(l_count,0);
for i=1:l_count
    l = l_min +(l_max-l_min)*rand(1,1);
    lambda_array(i) = 10^l;
end
    
end

function lambda_array = generate_fine_lambda(l_min, l_max, l_count)
%to generate the lambdas
lambda_array=zeros(l_count,0);
for i=1:l_count
    l = l_min +(l_max-l_min)*rand(1,1);
    lambda_array(i) = l;
end
    
end


function [W_cell, b_cell] = InitWB(K, m, d)
% to init W and B, p5

W_cell = cell(2, 1);
b_cell = cell(2, 1);

W1 = randn(m,d)*(1/sqrt(d)); 
W2 = randn(K,m)*(1/sqrt(m));

b1 = zeros(m,1);
b2 = zeros(K,1);

W_cell{1} = W1;
W_cell{2} = W2;

b_cell{1} = b1;
b_cell{2} = b2;
end
