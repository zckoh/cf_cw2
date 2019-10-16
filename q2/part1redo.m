% Load in the C/X call prices
opts = detectImportOptions('bs_call_prices.csv');
opts = setvartype(opts, opts.VariableNames(2:end) ,'double');
bs_price = readtable('bs_call_prices.csv', opts);

% Load in the TTM values
opts = detectImportOptions('ttm_call_prices.csv');
opts = setvartype(opts, opts.VariableNames(2:end) ,'double');
ttm_call_prices = readtable('ttm_call_prices.csv', opts);

% Load in the S/X values
opts = detectImportOptions('s_div_x.csv');
opts = setvartype(opts, opts.VariableNames(2:end) ,'double');
s_div_x = readtable('s_div_x.csv', opts);

% Remove first row and first column
bs_price(1,:) = [];
bs_price(:,1) = [];

ttm_call_prices(1,:) = [];
ttm_call_prices(:,1) = [];

s_div_x(1,:) = [];
s_div_x(:,1) = [];

% Prepare the dataset
X = [];
Y = [];
for i = 1:length(bs_price.Properties.VariableNames)
    % Store the X data inputs
    ttm_col = ttm_call_prices(:,i);
    ttm_col = ttm_col(~any(ismissing(ttm_col),2),:);
    
    s_div_x_col = s_div_x(:,i);
    s_div_x_col = s_div_x_col(~any(ismissing(s_div_x_col),2),:);
    
    comb_col = horzcat(table2array(s_div_x_col), table2array(ttm_col));
    X = vertcat(X, comb_col);
    
    % Store the target outputs
    col = bs_price(:,i);
    col = col(~any(ismissing(col),2),:);
    Y = vertcat(Y, table2array(col));
end

% Plot figure 4 for the entire dataset
figure();
scatter3(X(:,1),X(:,2),Y,'.');
hold on;
ylabel('T-t');
xlabel('S/K');
zlabel('C/K');
hold off;

% Split training and testing set for target input and outputs
rng(1);
[m,n] = size(X) ;
P = 0.70 ;
idx = randperm(m)  ;

training_vectors = X(idx(1:round(P*m)),:) ; 
testing_vectors = X(idx(round(P*m)+1:end),:) ;

training_labels = Y(idx(1:round(P*m)),:) ; 
testing_labels = Y(idx(round(P*m)+1:end),:) ;

% Estimate the mean and Sigma from training dataset
GMModel = fitgmdist(training_vectors,4 ,'SharedCovariance', true);

% Setup the design matrix
x1 = [];
x2 = [];
x3 = [];
x4 = [];
x56 = [];
x7 = ones(length(training_labels),1);
for i = 1:length(training_labels)
    x1 = vertcat(x1, mahalanobis(training_vectors(i,:),GMModel.mu(1,:),GMModel.Sigma));
    x2 = vertcat(x2, mahalanobis(training_vectors(i,:),GMModel.mu(2,:),GMModel.Sigma));
    x3 = vertcat(x3, mahalanobis(training_vectors(i,:),GMModel.mu(3,:),GMModel.Sigma));
    x4 = vertcat(x4, mahalanobis(training_vectors(i,:),GMModel.mu(4,:),GMModel.Sigma));
    
    x56 = vertcat(x56, X(i,:));
end

X_train = horzcat(x1,x2,x3,x4,x56,x7);

% Estimate the weights using Moore-Penrose Pseudoinverse
trained_weights = pinv(X_train) * training_labels;

% Setup the design matrix for testing split
x1 = [];
x2 = [];
x3 = [];
x4 = [];
x56 = [];
x7 = ones(length(testing_labels),1);
for i = 1:length(testing_labels)
    x1 = vertcat(x1, mahalanobis(testing_vectors(i,:),GMModel.mu(1,:),GMModel.Sigma));
    x2 = vertcat(x2, mahalanobis(testing_vectors(i,:),GMModel.mu(2,:),GMModel.Sigma));
    x3 = vertcat(x3, mahalanobis(testing_vectors(i,:),GMModel.mu(3,:),GMModel.Sigma));
    x4 = vertcat(x4, mahalanobis(testing_vectors(i,:),GMModel.mu(4,:),GMModel.Sigma));
    
    x56 = vertcat(x56, X(i,:));
end

X_test = horzcat(x1,x2,x3,x4,x56,x7);
Y_test = X_train * trained_weights;

% Plot figure 5a
% S/X, T-t, C/X
figure();
scatter3(X_train(:,5),X_train(:,6),Y_test, '.');
hold on;
ylabel('T-t');
xlabel('S/K');
zlabel('C');
hold off;

% Plot figure 5c
figure();
scatter3(X_train(:,5),X_train(:,6),Y_test-training_labels, '.');
colormap winter;
hold on;
ylabel('T-t');
xlabel('S/K');
zlabel('C Error');
hold off;

% [Xplt,Yplt] = meshgrid(X_test(:,5),X_test(:,6));
% 
% % xi = unique(X_test(:,5)) ; yi = unique(X_test(:,6)) ;
% % [Xplt,Yplt] = meshgrid(xi,yi) ;
% % Z = reshape(Y_test,size(Xplt)) ;
% % figure
% % surf(Xplt,Yplt,Z)
% 
% %%unstructured 
% dt = delaunayTriangulation(X_test(:,5),X_test(:,6)) ;
% tri = dt.ConnectivityList ;
% figure
% trisurf(tri,X_test(:,5),X_test(:,6),Y_test)


% % Load in the FTSE index and interest rate
% opts = detectImportOptions('FTSEOptionsData.xlsx', 'Sheet', 3);
% opts = setvartype(opts, opts.VariableNames(2:end) ,'double');
% FTSEData = readtable('FTSEOptionsData.xlsx', opts);
% 
% % Remove first row
% OptionsData(1,:) = [];
% FTSEData(1,:) = [];
% 
% % Rename the columns
% OptionsData.Properties.VariableNames{1} = 'Date';
% FTSEData.Properties.VariableNames = {'Date', 'FTSE100','r'};
% 
% 
% 
% % All the strike prices
% Strike_prices = [4000 4800 5200 5375 5600 5800 5900 6000  ...
% 6100 6150 6200 6250 6300 6350 6400 6450 6475  ...
% 6500 6525 6550 6575 6600 6625 6650 6675 6700  ...
% 6725 6750 6775 6800 6825 6850 6875 6900 6925  ...
% 6950 6975 7000 7025 7050 7075 7100 7125 7150  ...
% 7175 7200 7225 7250 7275 7300 7325 7350 7375  ...
% 7400 7425 7450 7475 7500 7525 7550 7575 7600  ...
% 7625 7650 7675 7700 7725 7750 7775 7800 7825  ...
% 7850 7875 7900 8000 8100 8200 8400 8600 8800  ...
% 9200 9600 10400];
% 
% % Extract date
% OptionsData.Date = datetime(OptionsData.Date,'InputFormat','dd/MM/yyyy');
% FTSEData.Date = datetime(FTSEData.Date,'InputFormat','dd/MM/yyyy');
% 
% %Tranform the data to timetable
% OptionsData = table2timetable(OptionsData);
% FTSEData = table2timetable(FTSEData);
% 
% % Find annualised volatility and store for later computation
% annual_vol = FTSEData(:,1);
% annual_vol = removevars(annual_vol,'FTSE100');
% annual_vol = addvars(annual_vol,'Volatility');
% 
% for t = range(1 + int(total_len/4) + 1, total_len + 1)
%     % Find the annualised volatility
%     pct_change = diff(FTSEData.FTSE100)./FTSEData.FTSE100(1:end-1,:);
%     annual_vol = std(pct_change) * sqrt(252);
% end
% 
% 
% for i = 1:length(OptionsData.Properties.VariableNames)
%     % Get the first column
%     col = table2array(OptionsData(:,i));
% 
%     % Find the timeseries info
%     total_len = length(col);
%     
%     for t = range(1 + int(total_len/4) + 1, total_len + 1)
%         % Find the annualised volatility
%         pct_change = diff(FTSEData.FTSE100)./FTSEData.FTSE100(1:end-1,:);
%         annual_vol = std(pct_change) * sqrt(252);
%     end
% end
