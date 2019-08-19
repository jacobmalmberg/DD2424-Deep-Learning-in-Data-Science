rng(401);
%load entire dataset
load('X.mat'); %Dxnlen
load('ys.mat'); %Nx1
y_one_hot=one_hot_encode_y(ys);


train_one_hot = one_hot_encode_y(trainY);


validation_one_hot = one_hot_encode_y(validationY);

K = max(unique(ys)); %18
d = 28; %28


number_of_networks = 2;
ConvNet_cell = cell(number_of_networks,1);


ConvNet_cell{1} = CreateNetwork(0.01, 401, 20, 5, 20, 3, sqrt(2/d), sqrt(2/20), sqrt(2/K)); %old init
ConvNet_cell{2} = CreateNetwork(0.001, 401, 40, 3, 40, 3, sqrt(2/d), sqrt(2/40), sqrt(2/K));

result_cell = cell(number_of_networks,4); %first column is validation_loss, second is accuracy, third is confusion matrix, fourth is convnet




%n_epochs = 110; %s.t. samples = 19700*110 = 2167000
n_epochs = 1200; %s.t. samples = 1200*1062 = 106200
%n_batch = 100;
n_batch = 59; 
N = 1062; % 59*18. where classes = 18. to account for imbalanced data
update_steps = (n_epochs*N)/n_batch % 10 * 197000 /100 = 1970 % 
n_update = 500;

% graph stuff
x_axis = 0:n_update:update_steps; % x-axeln.
x_axis = [x_axis update_steps]; %include last updatept too



for k=1:length(ConvNet_cell)
    %to test multiple convnets
    k
    ConvNet =  ConvNet_cell{k};
    MX_cell = Pre_compute_mx(trainX, ConvNet.F{1}); %precompute MX for layer 1
    
    %graph stuff
    train_loss = zeros(length(x_axis),1);
    validation_loss = zeros(length(x_axis),1);

    train_acc = zeros(length(x_axis),1);
    validation_acc = zeros(length(x_axis),1);

    confusion_matrices = cell(length(x_axis),1);
    
    %for momentum
    %initialize velocity to zero and momentum to something
    %if no momentum set momentum to 0, will ignore velocity, see
    %MiniBatchGD()
    ConvNet = InitVelocity(ConvNet);
    momentum = 0.9;
    
    
    t = 1; %update step
    save_iter=1; % for index counting

    for i=1:n_epochs
        %now sample
        [trainX_sampled, trainY_sampled, train_one_hot_sampled, MX_cell_sampled] = sample_randomly(trainX, trainY, train_one_hot, MX_cell);
        %shuffle it
        [trainX_sampled, trainY_sampled, train_one_hot_sampled, MX_cell_sampled] = shuffle_dataset(trainX_sampled, trainY_sampled , train_one_hot_sampled, MX_cell_sampled);

        for j=1:N/n_batch

            j_start = (j-1)*n_batch+1;
            j_end = j*n_batch;
            inds = j_start:j_end;
            X_batch = trainX_sampled(:, inds);
            training_one_hot_batch = train_one_hot_sampled(:, j_start:j_end);
            ConvNet = MiniBatchGD(X_batch, training_one_hot_batch, ConvNet, inds, MX_cell_sampled, momentum);

            if (t == 1) || (ismember(t,x_axis)) 

                train_loss(save_iter) = Compute_loss(trainX, train_one_hot, ConvNet)
                validation_loss(save_iter) = Compute_loss(validationX, validation_one_hot, ConvNet)
                [train_acc_iter, train_guesses, train_y] = ComputeAccuracy(trainX, trainY, ConvNet);
                train_acc(save_iter) = train_acc_iter
                [val_acc_iter, val_guesses, val_y] = ComputeAccuracy(validationX, validationY, ConvNet);
                validation_acc(save_iter) = val_acc_iter
                confusion_matrices{save_iter} = confusionmat(val_y, val_guesses);
                

                save_iter = save_iter +1;
            end

            t = t+1;

        end
    end
    
    result_cell{k,1} = validation_loss; % add validation loss
    result_cell{k,2} = validation_acc; % add validation acc 
    result_cell{k,3} = confusion_matrices{end}; % add last confusion matrix
    result_cell{k,4} = ConvNet; % add last confusion matrix
    
end


function ConvNet = MiniBatchGD(X, Y_one_hot, ConvNet, inds, MX_cell, momentum)
[grad_W, grad_F1, grad_F2] = ComputeGradients(X, Y_one_hot, ConvNet, inds, MX_cell);

if (momentum ~= 0)
    %use momentum 
    % https://cs231n.github.io/neural-networks-3/#sgd
    % no annealing, like a boss
    ConvNet.vW = momentum * ConvNet.vW - ConvNet.eta*grad_W;
    ConvNet.W = ConvNet.W + ConvNet.vW;
    
    ConvNet.vF1 = momentum * ConvNet.vF1 - ConvNet.eta*grad_F1;
    ConvNet.F{1} = ConvNet.F{1} + ConvNet.vF1;
    
    ConvNet.vF2 = momentum * ConvNet.vF2 - ConvNet.eta*grad_F2;
    ConvNet.F{2} = ConvNet.F{2} + ConvNet.vF2;
else
    %standard SGD
    ConvNet.F{1} = ConvNet.F{1} - ConvNet.eta*grad_F1;
    ConvNet.F{2} = ConvNet.F{2} - ConvNet.eta*grad_F2;
    ConvNet.W = ConvNet.W - ConvNet.eta*grad_W;
end

end



function MF = MakeMFMatrix(F, nlen)


[dd, k, nf] = size(F); %%dd = 28
dd;
MF = zeros((nlen-k+1)*nf, (nlen*dd));

%now convert F matrices into vectors
v_matrix = zeros(nf, dd*k); % VF, dd*k = numel(filter)
for i=1:nf
    v = F(:,:,i); %get a filter
    v = v(:)'; % turn it into a vec and transp
    v_matrix(i,:) = v; % into the matrix
end

%now put it into the MF matrix

v_len = dd*k; % length of vectorized filter

start_col=1; % which column to start writing to
for j=1:nf:size(MF,1)
    end_row = nf+j-1; % which row to stop writing to
    end_col = v_len+(start_col-1); % which col to stop writing to
    MF(j:end_row, start_col:end_col) = v_matrix;
    start_col = start_col +dd; % increment startcolumn

end
end

function MX = MakeMXMatrix(x_input, d, k, nf)
nlen = size(x_input,1)/d; % what if X is a giant matrix? hardcode? it never is!
%nlen = 19;

X_input = reshape(x_input, [d, nlen]); %make it into a matrix
%MX = zeros((nlen-k+1)*nf, (k*nf*d));
MX = sparse((nlen-k+1)*nf, (k*nf*d)); %sparse -> faster. NYCKEL

start_column=1;
for i=1:nf:size(MX,1)
    vecX = X_input(:,start_column:start_column+k-1); % get the vector belonging to the columns
    vecX = vecX(:)';  % turn it into a vec and transp
    repeated_vecX = repmat({vecX}, 1, nf); %repeat it, so it can be diagonalized
    diag_vecX = blkdiag(repeated_vecX{:}); % https://se.mathworks.com/matlabcentral/answers/39838-repeat-a-matrix-as-digonal-element-in-a-new-matrix
    MX(i:i+nf-1,:) = diag_vecX; % set it to diagonal
    start_column = start_column + 1; % increment start column, 1 since stride =1
end

end

function [grad_W, grad_F1, grad_F2] = ComputeGradients(X_batch, Y, ConvNet, inds, MX_cell)

% optimized with precomputed mx using sparse matrices.

N = size(Y,2); % datapoints in batch

% set to 0

grad_F1 = zeros(size(ConvNet.F{1}));
grad_F2 = zeros(size(ConvNet.F{2}));

[P_batch, X_batch1, X_batch2] = EvaluateClassifier(X_batch, ConvNet);

g_batch = -(Y-P_batch); %KxN
grad_W = 1/N * g_batch*X_batch2'; 

% propagate
g_batch = ConvNet.W'*g_batch; % n2*nlen2 x N
g_batch = g_batch .* (X_batch2 > 0); % n2*nlen2 x N, indicator function

%compute the gradient wrt the 2nd layer conv filters
[d,k,nf] = size(ConvNet.F{2}); %get dimensions for second layer

for j = 1:N
    g_j = g_batch(:,j);
    x_j = X_batch1(:,j);
    MX = MakeMXMatrix(x_j, d, k, nf);
    
    v = g_j'*MX;
    size(v);
    size(grad_F2);
    reshaped_v = reshape(v,[d,k,nf]); %reshape so it fits, page12
    grad_F2 = grad_F2 + 1/N * reshaped_v;
end

%propagate again

%make MF
nlen=19; %hardcode?
[~,k1,~] = size(ConvNet.F{1});

nlen1 = nlen-k1+1;

MF2=MakeMFMatrix(ConvNet.F{2}, nlen1);
g_batch = MF2'*g_batch;
g_batch = g_batch .* (X_batch1 > 0);

%compute the gradient wrt first layer conv layers
[d,k,nf] = size(ConvNet.F{1}); %get dimensions for first layer

for j = 1:N
    g_j = g_batch(:,j);
    x_j = X_batch(:,j);
    %MX = MakeMXMatrix(x_j, d, k, nf);
    MX = MX_cell{inds(j)};
    v = g_j' * MX;
    reshaped_v = reshape(v,[d,k,nf]); %reshape so it fits, page12
    grad_F1 = grad_F1 + 1/N * reshaped_v;
end
end


function loss = Compute_loss(X_batch, Ys_batch, ConvNet)
%Ys is one-hot encoded

[P, ~, ~] = EvaluateClassifier(X_batch, ConvNet);
sum_loss = sum(-log(dot(double(Ys_batch),P)));
N = size(X_batch,2); % Assuming the second column is N
loss = 1/N * sum_loss;

end

function [acc, I, y] = ComputeAccuracy(X,y,ConvNet)
%5
[P, ~, ~] = EvaluateClassifier(X, ConvNet);
[~,I] = max(P, [], 1); %argmax
N = size(y,1);
I=I';
acc = sum(y==I)/N;
end

function P = ComputeTestProbability(testX,ConvNet)
%5
[P, ~, ~] = EvaluateClassifier(testX, ConvNet);
end


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

function y_o_h = one_hot_encode_y(ys)
Y = ys == 1:max(ys);
y_o_h = Y';
end

function [shTrainX, shTrainY, shX_one_hot, sh_MX_cell] = shuffle_dataset(trainX, trainY, train_one_hot, MX_cell)
%since the dataset is not shuffled
%shuffles your mx_cell too

ny = size(trainX,2) ;
sh_MX_cell = cell(ny,1);
shuffle = randsample(1:ny,ny);
shTrainX = trainX(:,shuffle);
shX_one_hot = train_one_hot(:, shuffle);
shTrainY = trainY(shuffle);

for j=1:ny
    sh_MX_cell{j}=MX_cell{shuffle(j)};
end

end

function [shTrainX, shTrainY, shX_one_hot] = shuffle_dataset_only(trainX, trainY, train_one_hot)
%since the dataset is not shuffled
ny = size(trainX,2) ;
shuffle = randsample(1:ny,ny);
shTrainX = trainX(:,shuffle);
shX_one_hot = train_one_hot(:, shuffle);
shTrainY = trainY(shuffle);

end


function MX_cell = Pre_compute_mx(trainX, ConvNetF1)
% background 5 precompute mx

%put them all in a cell
N = size(trainX,2);
MX_cell = cell(N, 1);

[d,k,nf] = size(ConvNetF1); %get dimensions for first layer

for j = 1:N
    x_j = trainX(:,j);
    MX = MakeMXMatrix(x_j, d, k, nf);
    %MX_cell{j} = sparse(MX);
    MX_cell{j} = MX;
end
end

function [trainX_sampled, trainY_sampled, train_one_hot_sampled, MX_cell_sampled] = sample_randomly(trainX, trainY, train_one_hot, MX_cell)
%to account for unbalanced data
smallest_no = 59;% min(min(sum(trainY==trainY'))); %this should be 59..corresponding to class 18

total_no_samples = smallest_no*18; % %59 * 18 = 1062 samples in total

trainX_sampled = zeros(532, total_no_samples); 
trainY_sampled = zeros(total_no_samples,1); 
train_one_hot_sampled = zeros(18, total_no_samples); 

MX_cell_sampled = cell(total_no_samples,1);

%now go thorugh each class and get the same number of samples from each.
start_index = 1; %to put in correct place for trainX_sampled
for i=1:18
    all_indices = find(trainY==i); % get all teh indices for a class, to sample from
    
    sampled_indices = randsample(all_indices, smallest_no); %what indices to use
    sampled_classX = trainX(:, sampled_indices); %sample
    %sampled_classX = datasample(trainX(:, sampled_indices), smallest_no, 2); %sample
   
    label_list = trainY(all_indices);
    sampled_classY = label_list(1:smallest_no);
    sampled_class_one_hot = train_one_hot(:, sampled_indices);
    

    
    %now take the corresponding MX values
    for j=1:smallest_no
        MX_cell_sampled{start_index+j-1}=MX_cell{sampled_indices(j)};
    end
    
    %now put into datasetvector
    end_index = start_index + smallest_no-1;
    
    trainX_sampled(:, start_index:end_index) = sampled_classX;
    trainY_sampled(start_index:end_index) = sampled_classY;
    train_one_hot_sampled(:, start_index:end_index) = sampled_class_one_hot;
        
    start_index = end_index +1;
end
end


function ConvNet = CreateNetwork(eta, seed, n1, k1, n2, k2, sig1, sig2, sig3)
K = 18;
d = 28;
nlen = 19;

nlen1 = nlen-k1+1;
nlen2 = nlen1-k2+1;
fsize = n2*nlen2;

rng(seed);
ConvNet.F{1} = randn(d,k1,n1)*sig1;
ConvNet.F{2} = randn(n1,k2,n2)*sig2;
ConvNet.W = randn(K, fsize)*sig3;
ConvNet.eta = eta;
%rng(seed); % "reset it"

end

function ConvNet = InitVelocity(ConvNet)
%init the velocity to 0
ConvNet.vF1 = zeros(size(ConvNet.F{1}));
ConvNet.vF2 = zeros(size(ConvNet.F{2}));
ConvNet.vW = zeros(size(ConvNet.W));
end
