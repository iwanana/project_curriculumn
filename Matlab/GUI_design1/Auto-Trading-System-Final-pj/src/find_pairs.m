function pairs = find_pairs(dataset_handle)   %tickers is an string array contain all the tickers

tickers = fieldnames(dataset_handle.dataset); 
N = length(tickers);
pairs = {};

% filter on correct speed
delta = 0.1 * sqrt(1/390); 

% loop for every possible pair
for i = 1:(N-1)
    for j = (i+1):N

        %Price1 = dataset_handle.dataset.(tickers{i}.Table.Price);
        %Price2 = dataset_handle.dataset.(tickers{j}.Table.Price);
        Price1 = log(dataset_handle.dataset.(tickers{i}.Table.Price));
        Price2 = log(dataset_handle.dataset.(tickers{j}.Table.Price));

        [h, yb, xb] = coint(Price1,Price2);
        
        if (h == 1) && (yb > -1) && (yb < -delta) && (xb < 1) && (xb > delta)
            Stock1 = tickers{i};
            Stock2 = tickers{j};
            %%%%%%%%%%%%update above%%%%%%%%%%%%%%%%%%%
            new_pairs = {Stock1,Stock2};
            pairs = {pairs; new_pairs};
        end
        
    end
end
end


            
        
