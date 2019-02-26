function update_data(dataset_handle)
% update dataset in the handle structure
tickers = fieldnames(dataset_handle.dataset);

for i = 1:length(tickers)
    dataset_handle.dataset.(tickers{i}) = ...
        dataset_handle.dataset.(tickers{i}).update_live();
end