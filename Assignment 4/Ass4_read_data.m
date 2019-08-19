%reads in the data and stores it in the matrix

book_fname = 'Data/goblet_book.txt';

fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

book_chars = unique(book_data);
char_to_ind = CreateCharToInd(book_chars);
ind_to_char = CreateIndToChar(book_chars);
tic
X= EncodeInput(book_data);
toc
function char_to_ind = CreateCharToInd(book_chars)
char_to_ind=containers.Map('KeyType','char','ValueType','int32');
d = numel(book_chars);
for i= 1:d
    char_to_ind(book_chars(i)) = i;
end
end

function ind_to_char = CreateIndToChar(book_chars)
ind_to_char=containers.Map('KeyType','int32','ValueType','char');
d=numel(book_chars);
for i= 1:d
    ind_to_char(i) = book_chars(i);
end
end

function X = EncodeInput(all_names)
%same as in ass3
C = unique(all_names);
d = numel(C);
N = size(all_names,2);
n_len = max(strlength(all_names));

char_to_ind=CreateCharToInd(C);
X=zeros(d,N);
for i = 1:length(all_names)
    name_vec=zeros(d,1);
    char = all_names(i);    

    CharIndex=char_to_ind(char);
    name_vec(CharIndex) = 1;
    X(:,i) = name_vec;
    
end
            
        
        
end