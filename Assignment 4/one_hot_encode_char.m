function one_hot_char = one_hot_encode_char(index, K)
one_hot_char = zeros(K,1);
one_hot_char(index)=1;

end
