% run ExtractNames first
% generates X, trainX/Y, validation X/Y, testX (there is no test Y)
C = unique(cell2mat(all_names));
X = EncodeInputNames(all_names);
testX = Encode_test_names(all_names);
validation_inds = open_validation_inds();

train_inds = setdiff(1:size(X,2), validation_inds);

trainX = X(:, train_inds);
trainY = ys(train_inds);
validationX = X(:, validation_inds);
validationY = ys(validation_inds);

function X = EncodeInputNames(all_names)
C = unique(cell2mat(all_names));
d = numel(C); %28
N = size(all_names,2);
n_len = max(strlength(all_names)); %19

char_to_ind=CreateCharToInd(C, d);
X=zeros(d*n_len,N);
for i = 1:length(all_names)
    name_matrix=zeros(d,n_len);
    name = all_names{i};
    name_length = length(name);
    
    for j = 1:name_length
        CharIndex=char_to_ind(name(j));
        name_matrix(CharIndex,j) = 1;
    end
    name_vec = name_matrix(:);
    X(:,i) = name_vec;
    
end
            
        
        
end

function char_to_ind = CreateCharToInd(C, d)
char_to_ind=containers.Map('KeyType','char','ValueType','int32');
for i= 1:d
    char_to_ind(C(i)) = i;
end
end

function validation_inds = open_validation_inds()
f = fopen('Validation_Inds.txt');
data = textscan(f,'%s');
fclose(f);
validation_inds= str2double(data{1});
end

function testX = Encode_test_names(all_names)
%to encode the test names, must be lowercase

names = cell(5,1);
names{1} = 'malmberg';
names{2} = 'nystad';
names{3} = 'hotti';
names{4} = 'singh';
names{5} = 'selmer';


C = unique(cell2mat(all_names));
d = numel(C); %28
N = size(names,1);
n_len = max(strlength(all_names)); %19

char_to_ind=CreateCharToInd(C, d);
testX=zeros(d*n_len,N);

for i = 1:length(names)
    name_matrix=zeros(d,n_len);
    name = names{i};
    name_length = length(name);
    
    for j = 1:name_length
        CharIndex=char_to_ind(name(j));
        name_matrix(CharIndex,j) = 1;
    end
    name_vec = name_matrix(:);
    testX(:,i) = name_vec;
    
end
            
      

end