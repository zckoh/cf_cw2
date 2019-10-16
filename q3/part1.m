StockData = readtable('S&P500.csv','ReadVariableNames',true);

%Ensure the data is in correct data type
if isnumeric(StockData.Open) == false
    Open =cellfun(@str2double,StockData.Open);
    High = cellfun(@str2double,StockData.High);
    Low = cellfun(@str2double,StockData.Low);
    Close = cellfun(@str2double,StockData.Close);
    Volume = cellfun(@str2double,StockData.Volume);
else
    Open = StockData.Open;
    High = StockData.High;
    Low = StockData.Low;
    Close = StockData.Close;
    Volume =  StockData.Volume;
end
Date = StockData.Date;

%Tranform the data to timetable
StockData_TimeTable = timetable(Date,Open,High,Low,Close,Volume);
%Check for missing Data
%Fill the missing data with previous value
if any(any(ismissing(StockData_TimeTable)))==true
    StockData_TimeTable = fillmissing(StockData_TimeTable,'previous');
end

%View the data
plot(StockData_TimeTable.Date,StockData_TimeTable.Close);
ylabel('Index');
xlabel('Timeline');
grid on

% Extract 4 years of data for estimating and fitting
tr = timerange('2014-01-02' , '2017-12-31');
StockData_TimeTable_4Years = StockData_TimeTable(StockData_TimeTable.Date(tr),:);

% clear unwanted variables
clear ('Open','Close','Date','High','Low','Volume','tr','StockData');

% Initialize the AR(3) model with no constant term
AR_order3 = arima('ARLags',1:3,'Constant',0);

[AR_order3_est,~,Loglikehood] = estimate(AR_order3,StockData_TimeTable_4Years.Close);

rng(1); % For reproducibility

%View the predictive vallue
residual4years = infer(AR_order3_est,StockData_TimeTable_4Years.Close);
prediction = StockData_TimeTable_4Years.Close + residual4years;
figure
plot(StockData_TimeTable_4Years.Date,StockData_TimeTable_4Years.Close);
hold on
plot(StockData_TimeTable_4Years.Date,prediction);
ylabel('Index');
xlabel('Timeline');
legend('Actual','Third-Order AR','Location','best');
grid on
hold off

residual_var = var(residual4years);

% Save the stockData, AR model and the residuals
save('part1', 'StockData_TimeTable', 'AR_order3_est', 'residual4years')


