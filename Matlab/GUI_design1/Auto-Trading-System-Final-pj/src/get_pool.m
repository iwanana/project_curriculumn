function [Pool] = get_pool(industry)
%GET_POOL Summindustryry of this function goes here
%   Detailed explanation goes here
universe = readtable('tickerbase.dat','Delimiter',',');
if industry==1
    index= [1:320];
elseif industry==2
    index=[321:692];
elseif industry==3
    index=[693:927];
elseif industry==4
    index=[928:1749];
elseif industry==5
    index=[1750:2064];
elseif industry==6
    index=[2065:3101];
elseif industry==7
    index=[3102:3928];
elseif industry==8
    index=[3929:4072];
elseif industry==9
    index=[4073:4353];
elseif industry==10
    index=[4354:4972];
else
    index=[4973:5094];
end
Pool = universe{index,1};
end

