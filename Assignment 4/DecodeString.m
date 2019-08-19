function string = DecodeString(Y, ind_to_char)
string=[];
[~,I] = max(Y, [], 1); %argmax

for i=1:length(I)
    string = [string ind_to_char(I(i))];
end
end