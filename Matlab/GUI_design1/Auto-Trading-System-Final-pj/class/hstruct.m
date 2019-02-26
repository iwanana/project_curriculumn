classdef hstruct < handle
    % Handle structure class
    properties
        dataset
    end
    
    methods
        function obj = hstruct(s)
			obj.dataset = s;
        end 
    end
end

