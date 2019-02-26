function [signals, shares] = pair_signal(dataset_handle, pairs, ptable_handle)
% % get updated stock pair data through dataset_handle
% % paris are cell array of cell array - e.g. {{a1,b1}, {a2,b2}}
% 
% %%% Following code are sample code for MA5 strategy in HW8 %%%
% 
% % stocks under consideration
% tickers = fieldnames(dataset_handle.dataset);
% 
% % memory allocation
% N = length(tickers);
% signals = zeros(N,1);
% shares = ones(N,1);
% 
% % create positions structure
% positions = struct();
% for i = 1:N
%     positions.(tickers{i}) = 0;
% end
% 
% % get current positions
% M = size(ptable_handle.Data,1);
% for i = 1:M
%     positions.(ptable_handle.Data{i,1}) = ptable_handle.Data{i,2};
% end
% 
% % parameter of the strategy
% delta = 0.02;
% 
% for i = 1:N
%     price = dataset_handle.dataset.(tickers{i}).Table.Price;
%     MA5_now = mean(price(end-4:end));
%     MA5_last = mean(price(end-5:end-1));
% 
%     if (MA5_now > MA5_last + delta) && positions.(tickers{i}) == 0
%         signals(i) = 1;
%     elseif (MA5_now < MA5_last - delta) && positions.(tickers{i}) == 1
%         signals(i) = -1;
%     end
%     
% end
tickers = fieldnames(dataset_handle.dataset);

% N is the # of pairs
N = length(pairs);
Total_invest = 1000000;
Pair_invest = Total_invest/N;
signals = {};
shares = {};
positions = struct();
for i = 1:N
    positions.(tickers{i}) = 0;
end

% % get current positions
M = size(ptable_handle.Data,1);
for i = 1:M
    positions.(ptable_handle.Data{i,1}) = ptable_handle.Data{i,2};
end
% parameter of the strategy
% delta = 0.02;
for i = 1:N
    pairs_now = pairs{i};
    Ticker_X = pairs_now{1};
    Ticker_Y = pairs_now{2};
    Price_X = dataset_handle.dataset.(Ticker_X).Table.Price;
    Price_Y = dataset_handle.dataset.(Ticker_Y).Table.Price;
    reg = fitlm(Price_X,Price_Y);
    err = reg.Residuals.Raw;
    sigma = reg.MSE;
    spread = err(end);
    score = spread/sigma;
    if score > 2 && positions.(Ticker_X) == 0 && positions.(Ticker_Y) == 0
        %  long X and short Y
        new_signal = {1,-1};
    elseif score < -2 && positions.(Ticker_X) == 0 && positions.(Ticker_Y) == 0
        %  short X and long Y
        new_signal = {-1,1};
    elseif abs(score) < 1  %&&(~(positions.(Ticker_X) == 0) || ~(positions.(Ticker_Y) == 0))
            % close the position 
        if (~(positions.(Ticker_X) == 0) || ~(positions.(Ticker_Y) == 0))
            new_signal = {-(positions.(Ticker_X)),-(positions.(Ticker_Y))};
        else
            %(positions.(Ticker_X) == 0 && positions.(Ticker_Y) == 0)
            % keep the position to be closed
            new_signal = {0,0}; 
        end         
    elseif abs(score) > 4 %&&(~(positions.(Ticker_X) == 0) || ~(positions.(Ticker_Y) == 0))
        if ~(positions.(Ticker_X) == 0) || ~(positions.(Ticker_Y) == 0)
            new_signal = {-(positions.(Ticker_X)),-(positions.(Ticker_Y))};  
        else %(positions.(Ticker_X) == 0 && positions.(Ticker_Y) == 0)
            new_signal = {0,0};
        end
    else  %just hold the position when -2 <score<-1 
        new_signal = {(positions.(Ticker_X)),(positions.(Ticker_Y))};
    end
    signals = {signals;new_signal};
    Beta = reg.Coefficients{'x1','Estimate'};
    Hedge_ratio = Beta*Price_X(end)/Price_Y(end);
    Y_invest = Pair_invest / (1+Hedge_ratio);
    X_invest = Pair_invest - Y_invest;
    new_share = {quantity(Price_X(end),X_invest),quantity(Price_Y(end),Y_invest)};
    shares = {shares;new_share};
end

    