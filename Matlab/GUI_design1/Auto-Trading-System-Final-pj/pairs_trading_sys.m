function varargout = pairs_trading_sys(varargin)
% System MATLAB code for System.fig
%      System, by itself, creates a new System or raises the existing
%      singleton*.
%
%      H = System returns the handle to a new System or the handle to
%      the existing singleton*.
%
%      System('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in System.M with the given input arguments.
%
%      System('Property','Value',...) creates a new System or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before System_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to System_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help System

% Last Modified by GUIDE v2.5 01-May-2018 14:24:58

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @System_OpeningFcn, ...
                   'gui_OutputFcn',  @System_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- OpeningFcn & OutputFcn
function System_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to System (see VARARGIN)

% Choose default command line output for System
handles.output = hObject;

% Create handle structure and add the handle to handles
% handles.dataset_handle = hstruct(struct());

% Update handles structure
guidata(hObject, handles);
function varargout = System_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- CreateFcn
function ticker_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ticker_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
function shares_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to shares_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
function port_pos_table_CreateFcn(hObject, eventdata, handles)
hObject.ColumnName = {'Ticker','Position','Mkt Price','Avg Cost','P&L'};


% --- Order Module
function buy_button_Callback(hObject, eventdata, handles)
% market order buy - could be further modified to mkt/lmt
% see if order has been make
symbol = handles.ticker_edit.String;
shares = str2double(handles.shares_edit.String);
IBMatlab('action','BUY', 'symbol',symbol,'quantity',shares,'type','MKT')
update_table(handles.port_pos_table)
function sell_button_Callback(hObject, eventdata, handles)
% market order sell - could be further modified to mkt/lmt
symbol = handles.ticker_edit.String;
shares = str2double(handles.shares_edit.String);
IBMatlab('action','SELL', 'symbol',symbol,'quantity',shares,'type','MKT')
update_table(handles.port_pos_table)
function ticker_edit_Callback(hObject, eventdata, handles)
function shares_edit_Callback(hObject, eventdata, handles)


% --- System Module
function sys_start_button_Callback(hObject, eventdata, handles)
% Get current portfolio
update_table(handles.port_pos_table)
function sys_stop_button_Callback(hObject, eventdata, handles)
hObject.Value = 0;
function Trade_button_Callback(hObject, eventdata, handles)
% Get stock pool

% Pool = {'AAPL', 'AMZN', 'BABA', 'BAC', 'FB', 'GE', 'NVDA', 'TSLA'};
% Pool = get_pool();
%Pool = {'AAPL', 'AMZN'};
industry = handles.popupmenu1.Value;
Pool = get_pool(industry);




% Create dataset handle
dataset_handle = hstruct(struct());

% Get historical data
historical_data(dataset_handle, Pool)

% Get pairs
pairs = find_pairs(dataset_handle);

% Switch on the strategy
handles.sys_stop_button.Value = 1;

% Loop until system stops
while handles.sys_stop_button.Value
    
    
    % update_data(dataset_handle)
    %%% ---------------------------------------------------------- %%%    
    % update new live data for pairs
    update_data(dataset_handle, paris)
    
    % get signal for each stock
    [signals, shares] = pair_signal(dataset_handle, pairs, handles.port_pos_table);
    
    % make orders
    execute_order(pairs, signals, shares)
    %%% ---------------------------------------------------------- %%%    
    
    % update port_pos_table
    update_table(handles.port_pos_table)
    
    % update P&L chart
    update_chart(handles.port_pos_table)
    
    pause(1)
end


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
