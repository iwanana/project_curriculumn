%% Data class designed by Yu Jiabo
%  Properties: stock symbol; table of historical data
%  Methods: update data

classdef Data
    
    properties
        Symbol
        Table
    end
    
    
    methods
        function obj = Data(Symbol)
            IBdat = IBMatlab('action','history','symbol',Symbol,'barSize','1 min','DurationValue',2,'useRTH',1);
            % IBdat = IBMatlab('action','history','symbol',Symbol,'barSize','1 day','DurationValue',252,'useRTH',1)
            % IBdat = IBMatlab('action','history', 'LocalSymbol','SPM8', 'SecType','FUT', 'Exchange','GLOBEX','barSize','1 min');
            obj.Symbol = Symbol;
            obj.Table = Data.generateTable(IBdat);
        end
        
        function obj = update_live(obj)     % This method is used to retrieve live data and update Data
            newIBdat = IBMatlab('action','query', 'symbol',obj.Symbol,'useRTH',0);
            % newIBdat = IBMatlab('action','query', 'LocalSymbol','SPM8', 'SecType','FUT', 'Exchange','GLOBEX');            
            newTable = Data.generateTable(newIBdat);
            obj.Table = unique(vertcat(obj.Table,newTable));
        end
        
        function obj = update_hist(obj)
            % This method is used to update data for a long time gap
            
            newIBdat = IBMatlab('action','history','symbol',obj.Symbol,'barSize','1 min','useRTH',0);
            newTable = Data.generateTable(newIBdat);
            obj.Table = unique(vertcat(obj.Table,newTable));
        end 
        
        function obj = retime(obj)
            obj.Table = retime(obj.Table,'minutely','linear');
        end
                
        function display(obj)
            % default display when no semicolon at the end of statement
            disp(obj)
            fprintf('\nHead of historical data timetable:\n')
            disp(head(obj.Table))
            fprintf('\nTail of historical data timetable:\n')
            disp(tail(obj.Table))
        end
        
        
    end
    
    
    methods (Access = private, Static)
        function Timetable = generateTable(IBdat)
            % Convert IBMatlab type data to timetable
            
            try % historical data
                Datetime = datetime(datestr(IBdat.dateNum));
                Price = IBdat.WAP';
                
            catch ME
                if ~isempty(ME) % live data
                    Datetime = datetime(IBdat.dataTime);
                    Price = IBdat.lastPrice;
%                     Price = 0.5 * (IBdat.askPrice + IBdat.bidPrice);

                end
            end
              
            Timetable = timetable(Datetime,Price);
        end
    
    end
end

