function MF = MakeMFMatrix(F, nlen)

[dd, k, nf] = size(F); %%dd = 28
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
    j;
    nf;
    end_row = nf+j-1; % which row to stop writing to
    end_col = v_len+(start_col-1); % which col to stop writing to
    MF(j:end_row, start_col:end_col) = v_matrix;
    start_col = start_col +dd; % increment startcolumn

end
end