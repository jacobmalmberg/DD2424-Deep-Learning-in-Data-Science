function xnext = sample_char(p, K)
cp = cumsum(p);
a = rand;
ixs = find(cp -a >0);
ii = ixs(1);
xnext = one_hot_encode_char(ii, K);

end
