
global j
for j=1:10
    dosome()
end

function dosome()
global j
disp(j)
end
