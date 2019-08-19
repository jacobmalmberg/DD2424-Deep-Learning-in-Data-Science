A = reshape(1:20,4,5) ; % example matrix
c = [2 3] ; % columns to shuffle
n = size(A,1);
for k = c
   A(:,k) = A(randperm(n),k) ;
end


k = reshape(1:20,4,5) ; % example matrix
%Y = a_one_hot(:,1:5);
k
Y
ny =size(k,2) ;
shuffle = randsample(1:ny,ny);
k_shuffle = k(:,shuffle);
Y_shuffle = Y(:,shuffle);
k_shuffle
Y_shuffle
ny =size(k,2) ;
shuffle = randsample(1:ny,ny);
k_shuffle = k(:,shuffle);
Y_shuffle = Y(:,shuffle);
k_shuffle
Y_shuffle
