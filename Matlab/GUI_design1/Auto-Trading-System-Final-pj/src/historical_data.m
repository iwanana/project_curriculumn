function historical_data(dataset_handle, Pool)
% Get historical data for stock tickers in Pool and store data through handle
% dataset_handle - handle for a handle structure
% Pool - cell array

for i = 1:length(Pool)
    dataset_handle.dataset.(Pool{i}) = Data(Pool{i});
end
