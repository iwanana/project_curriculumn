function update_chart(ptable_handle)

% portfolio P&L
M = size(ptable_handle.Data,1);
PNL = 0;
for i = 1:M
    PNL = PNL + ptable_handle.Data{i,5};
end

hold on;
scatter(ptable_handle.UserData, PNL,[],'b')
