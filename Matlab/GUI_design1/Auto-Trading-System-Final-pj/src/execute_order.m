function execute_order(pairs, signals, shares)
% Make orders to TWS according to the signals

%tickers = fieldnames(dataset_handle.dataset);
%N = length(tickers);
% for i = 1:N
%     if signals(i) == 1
%         IBMatlab('action','BUY', 'symbol',tickers{i},'quantity',shares(i),'type','MKT')
%     elseif signals(i) == -1
%         IBMatlab('action','SELL', 'symbol',tickers{i},'quantity',shares(i),'type','MKT')
%     end
% end
N = length(signals);
for i=1:N
    pair_now = pairs{i};
    signal_now = signals{i};
    shares_now = shares{i};
    for j = 1:2
        if signal_now{j} == 1
            IBMatlab('action','BUY', 'symbol',pair_now{j},'quantity',shares_now{j},'type','MKT');
        elseif signals{1} == -1
            IBMatlab('action','BUY', 'symbol',pair_now{j},'quantity',shares_now{j},'type','MKT');
        end
    end
end

    



