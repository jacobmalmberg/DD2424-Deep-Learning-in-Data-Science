% to generate tweets
load('bonusData.mat') %to get X
load('ass4_trump_res_cell1.mat');
%1st is loss, 2nd is RNN, 3rd is hprev, 
%4th is bestRNN, 5h is besthprev
%, 6th is cell of  tweets

RNN =result_cell{1,2};
hprev = result_cell{1,3};

done = false;
goals = {'loser', 'sad', 'fake', 'idiot', 'news', 'huge', 'liar', 'obama', 'great', 'hillary', 'china'};
goals = {'fake news', 'sad', 'loser'};
x0 = X{1};
no_dp = size(X,1);
%hprev = zeros(size(hprev));



while ~done
    tweet_length = randi([20 140]);
    %x0 = X{randi([1 no_dp])};
    x0 = [];
    while (isempty(x0)) %since some datapoints are fucked
        x0 = X{randi([1 no_dp])};
    end
    Y = synt_text(RNN, hprev, x0(:,1), tweet_length);
    string = DecodeString(Y, ind_to_char_bonus);
    if contains(string, goals, 'IgnoreCase', true)
        done = true;
        string
    end
end

    