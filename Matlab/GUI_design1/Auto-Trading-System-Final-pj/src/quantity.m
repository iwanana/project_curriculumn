function [share] = quantity(price,invest)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    share = fix(captial/(price*100));
    share = share * 100;
end

