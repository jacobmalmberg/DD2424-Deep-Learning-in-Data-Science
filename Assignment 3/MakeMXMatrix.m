function MX = MakeMXMatrix(x_input, d, k, nf)
nlen = size(x_input,1)/d; % what if X is a giant matrix? hardcode? it never is!
X_input = reshape(x_input, [d, nlen]); %make it into a matrix
MX = zeros((nlen-k+1)*nf, (k*nf*d));

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