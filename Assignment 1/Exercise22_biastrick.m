% [trainX, training_one_hot, trainY] = LoadBatch('data_batch_1.mat');
% [validationX, validation_one_hot, validationY] = LoadBatch('data_batch_2.mat');
[testX, test_one_hot, testY] = LoadBatch('test_batch.mat');
% 
[aX, a_one_hot, aY] = LoadBatch('data_batch_1.mat');
[bX, b_one_hot, bY] = LoadBatch('data_batch_2.mat');
[cX, c_one_hot, cY] = LoadBatch('data_batch_3.mat');
[dX, d_one_hot, dY] = LoadBatch('data_batch_4.mat');
[eX, e_one_hot, eY] = LoadBatch('data_batch_5.mat');

validationX = aX(:,1:1000);
validation_one_hot = a_one_hot(:,1:1000);

trainX = [aX(:,1001:10000) bX cX dX eX];
training_one_hot =[a_one_hot(:,1001:10000) b_one_hot c_one_hot d_one_hot e_one_hot];




%sanity check #3
% trainX = trainX(:,1:100);
% training_one_hot = training_one_hot(:,1:100);
% 
% validationX = validationX(:,1:100);
% validation_one_hot = validation_one_hot(:,1:100);
% 
% testX = testX(:,1:100);
% test_one_hot = test_one_hot(:,1:100);
% testY= testY(1:100);

d = 3072;
N = size(trainX,2);
K = 10;
std_dev = 0.01;
mean = 0;

% epoch etc settings
rng(400); % seed
n_batch = 100;
eta = .001;
n_epochs = 40;
lambda = 0.1;

%init matrices for cost and loss per epoch
train_cost = zeros(n_epochs,1);
validation_cost = zeros(n_epochs,1);

train_loss = zeros(n_epochs,1);
validation_loss = zeros(n_epochs,1);
%2 init W and B

W = randn(K,d)*std_dev; %correct?
b = randn(K,1)*std_dev;

%3 test

%J = ComputeCost(trainX, training_one_hot, W, lambda)

% numerical tests
eps=1e-9;
% http://cs231n.github.io/neural-networks-3/#gradcheck
% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(1:20,1), training_one_hot(:,1), W(:, 1:20), b , lambda, 1e-6);
% P = EvaluateClassifier(trainX(1:20,1), W(:, 1:20), b);
% %size(trainX(1:20,1))
% [grad_W, grad_B] = ComputeGradients(trainX(1:20,1), training_one_hot(:,1), P, W(:,1:20), lambda);
% grad_B_rel_error= abs(grad_B - ngrad_b) ./ max(eps, abs(grad_B) + abs(ngrad_b));
% grad_W_rel_error = abs(grad_W - ngrad_W) ./ max(eps, abs(grad_W) + abs(ngrad_W));
% sum(1e-4 < grad_B_rel_error)

%[ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(:,1:100), training_one_hot(:, 1:100), W, b , lambda, 1e-6);

% now, do the bias trick

trainX = [trainX; ones(1,N)];
validationX = [validationX; ones(1,size(validation_one_hot,2))];
testX = [testX; ones(1,size(test_one_hot,2))];

W = [W b];


%P = EvaluateClassifier(trainX(:,1:100), W);
%size(trainX(1:20,1))



%grad_W = ComputeGradients(trainX(:,1:100), training_one_hot(:, 1:100), P, W, lambda);
%grad_B_rel_error= abs(grad_W(:, 3073) - ngrad_b) ./ max(eps, abs(grad_W(:, 3073)) + abs(ngrad_b));
%grad_W_rel_error = abs(grad_W(:,1:3072) - ngrad_W) ./ max(eps, abs(grad_W(:,1:3072)) + abs(ngrad_W));
% sum(1e-4 < grad_B_rel_error)

%7

for i=1:n_epochs
    % 2.1g
    % https://se.mathworks.com/matlabcentral/answers/315631-how-to-permute-the-columns-in-a-matrix-in-random-way
%     ny =size(trainX,2) ;
%     shuffle = randsample(1:ny,ny) ;
%     trainX = trainX(:,shuffle);
%     training_one_hot = training_one_hot(:, shuffle);    
   
    for j=1:N/n_batch

        
        j_start = (j-1)*n_batch+1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        X_batch = trainX(:, inds);
        %Y_batch = trainY(:, inds);
        training_one_hot_batch = training_one_hot(:, j_start:j_end);
        W = MiniBatchGD(X_batch, training_one_hot_batch, W, lambda, eta);
    end
    
    % 2.1d
    %eta = eta*0.9;
    train_cost(i) = ComputeCost(trainX, training_one_hot, W, lambda);
    validation_cost(i) = ComputeCost(validationX, validation_one_hot, W, lambda);
    
    train_loss(i) = ComputeLoss(trainX, training_one_hot, W, b);
    validation_loss(i) = ComputeLoss(validationX, validation_one_hot, W, b);
    
end
    
%accuracy, create plot, visualize W
acc = ComputeAccuracy(testX,testY,W)
lambda
eta
x_axis = 1:n_epochs;
name_array = ['Cost_lambda=0_eta=0001'];
make_plot(x_axis, train_cost, validation_cost, 'epoch', 'cost', name_array)

name_array = ['Loss_lambda=0_eta=0001'];
make_plot(x_axis, train_loss, validation_loss, 'epoch', 'loss', name_array)



function make_plot(x_axis, train_vec, validation_vec, x_label, y_label, name_array)
% to make plots like fig3/4
figure;
plot(x_axis, train_vec, x_axis, validation_vec);
legend('training','validation')
ylabel(y_label); 
xlabel(x_label);
path = "Result_Pics/Ex22/";
name = name_array;
%title(name)
saveas(gcf, path + name +".png") %gcf = current
end


function [X, Y, y] = LoadBatch(filename)
%pt 1
A = load(filename);
X = single(A.data')/255;   %convert values to 0-1  
y = double(A.labels +1); % adjust for matrix 1 indexing
%Y = bsxfun(@eq, y(:), 1:max(y)); % https://stackoverflow.com/questions/38947948/how-can-i-hot-one-encode-in-matlab
Y = y == 1:max(y);
Y= double(Y');
end

function s = EvaluateClassifier(X, W)
%3
%lect 2 slide 80/82
s = W*X;
end

function J = ComputeCost(X, Y, W, lambda)
%4, lect2 slide 89-93
%Y is one-hot encoded
N = size(X,2); % Assuming the second column is N
W_sumsqr=sumsqr(W(:,1:3072)); % https://se.mathworks.com/help/deeplearning/ref/sumsqr.html

s = EvaluateClassifier(X, W);
sy = s.*Y; % sy, the score for the correct class
%[sy_vec,~] = max(sy, [], 1) %argmax, get sy into a 1xN vector
[~,~,sy_vec] = find(sy);
sy_vec=sy_vec';
% ones_vec = ones(K,1); % to turn sy into column matrix
% sy_matrix = ones_vec*sy; % matrix with sy on all rows
l = max(0, s-sy_vec+1);% loss, lect 2 slide91 93

%now sum it up, but only for the wrong classes, see slides
l=sum(sum(l-Y)); %twice since its a matrix

J = 1/N * l + lambda*W_sumsqr; 

end

function loss = ComputeLoss(X, Y, W, lambda)
%4, lect2 slide 89-93
%Y is one-hot encoded
N = size(X,2); % Assuming the second column is N
s = EvaluateClassifier(X, W);
sy = s.*Y; % sy, the score for the correct class
%[sy_vec,~] = max(sy, [], 1) %argmax, get sy into a 1xN vector
[~,~,sy_vec] = find(sy);
sy_vec=sy_vec';
% ones_vec = ones(K,1); % to turn sy into column matrix
% sy_matrix = ones_vec*sy; % matrix with sy on all rows
l = max(0, s-sy_vec+1);% loss, lect 2 slide91 93

%now sum it up, but only for the wrong classes, see slides
l=sum(sum(l-Y)); %twice since its a matrix

loss = 1/N * l;

end

function acc = ComputeAccuracy(X,y,W)
%5
P = EvaluateClassifier(X,W);
[~,I] = max(P, [], 1); %argmax
N = size(y,1);
I=I';
acc = sum(y==I)/N;
end


function grad_W = ComputeGradients(X, Y, s, W, lambda)
%6
%X, Y are minibatches (DxN)
N = size(Y,2);
sy = s.*Y; % sy, the score for the correct class
%[sy_vec,~] = max(sy, [], 1) %argmax, get sy into a 1xN vector
[~,~,sy_vec] = find(sy); %argmax
sy_vec=sy_vec';
l = max(0, s-sy_vec+1);% loss, lect 2 slide91 93
s_wo_correct_class = l-Y;

non_zero = sum(0 < s_wo_correct_class); % 3

binary_s = s_wo_correct_class~=0; % 4

five = -non_zero.*Y;

six = binary_s + five;

seven = six * X';
regularization = 2*lambda*W(:,1:3072); %dont include the bias
reg = [regularization zeros(size(W,1),1)]; %zeros at bias
grad_W = 1/N * seven + reg;
end

function Wstar = MiniBatchGD(X, Y, W, lambda, eta)
s = EvaluateClassifier(X,W);
grad_W = ComputeGradients(X, Y, s, W, lambda);
Wstar = W - eta*grad_W;
end

function vis(K, W_plot)
mt = [];
for i=1:K
  im = reshape(W_plot(i, :), 32, 32, 3);
  s_im{i} = (im-min(im(:)))/(max(im(:))-min(im(:)));
  s_im{i} = permute(s_im{i}, [2, 1, 3]);
  mt = [mt s_im{i}];
    
  
  
end
figure;
montage(s_im);
%montage(mt);
end
