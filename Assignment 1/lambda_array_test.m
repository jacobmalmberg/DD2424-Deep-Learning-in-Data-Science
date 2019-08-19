l_min = -5;
l_max = -1;
l_count=15;
%to generate the lambdas
lambda_array=zeros(l_count,1);
for i=1:l_count
    l = l_min +(l_max-l_min)*rand(1,1);
    lambda_array(i) = 10^l
end
%lambda_array