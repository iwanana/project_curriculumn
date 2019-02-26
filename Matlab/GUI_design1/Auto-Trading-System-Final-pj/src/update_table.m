function update_table(table_handle)
% Require portfolio
Portfolio = IBMatlab('action','portfolio');

% Allocate memory
table_handle.Data = cell(length(Portfolio),5);

for i = 1:length(Portfolio)
    table_handle.Data{i,1} = Portfolio(i).symbol;
    table_handle.Data{i,2} = Portfolio(i).position;
    table_handle.Data{i,3} = Portfolio(i).marketPrice;
    table_handle.Data{i,4} = Portfolio(i).averageCost;
    table_handle.Data{i,5} = (Portfolio(i).marketPrice - ...
        Portfolio(i).averageCost) * Portfolio(i).position;     
end

% Record time
table_handle.UserData = datetime();