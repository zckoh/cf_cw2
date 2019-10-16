% Load in the C/X call prices
opts = detectImportOptions('c_div_x.csv');
opts = setvartype(opts, opts.VariableNames(2:end) ,'double');
c_div_x = readtable('c_div_x.csv', opts);

% Load in the TTM values
opts = detectImportOptions('ttm_call_prices.csv');
opts = setvartype(opts, opts.VariableNames(2:end) ,'double');
ttm_call_prices = readtable('ttm_call_prices.csv', opts);

% Load in the S/X values
opts = detectImportOptions('s_div_x.csv');
opts = setvartype(opts, opts.VariableNames(2:end) ,'double');
s_div_x = readtable('s_div_x.csv', opts);

% Load in the K values
opts = detectImportOptions('k_df.csv');
opts = setvartype(opts, opts.VariableNames(2:end) ,'double');
k = readtable('k_df.csv', opts);

% Load in the true dcds values
opts = detectImportOptions('dcds.csv');
opts = setvartype(opts, opts.VariableNames(2:end) ,'double');
dcds_true = readtable('dcds.csv', opts);

% Remove first row and first column
c_div_x(1,:) = [];
c_div_x(:,1) = [];

ttm_call_prices(1,:) = [];
ttm_call_prices(:,1) = [];

s_div_x(1,:) = [];
s_div_x(:,1) = [];

k(1,:) = [];
k(:,1) = [];

dcds_true(1,:) = [];
dcds_true(:,1) = [];


% Prepare the dataset
X = [];
Y = [];
dcds_array = [];
for i = 1:length(c_div_x.Properties.VariableNames)
    % Store the X data inputs
    ttm_col = ttm_call_prices(:,i);
    ttm_col = ttm_col(~any(ismissing(ttm_col),2),:);
    
    s_div_x_col = s_div_x(:,i);
    s_div_x_col = s_div_x_col(~any(ismissing(s_div_x_col),2),:);
    
    comb_col = horzcat(table2array(s_div_x_col), table2array(ttm_col));
    X = vertcat(X, comb_col);
    
    % Store the target outputs
    col = c_div_x(:,i);
    col = col(~any(ismissing(col),2),:);
    Y = vertcat(Y, table2array(col));
    
    dcds_col = dcds_true(:,i);
    dcds_col = dcds_col(~any(ismissing(dcds_col),2),:);
    dcds_array = vertcat(dcds_array, table2array(dcds_col));
end

% Split training and testing set for target input and outputs
rng(1);
[m,n] = size(X) ;
P = 0.70 ;
idx = randperm(m)  ;

training_vectors = X(idx(1:round(P*m)),:) ; 
testing_vectors = X(idx(round(P*m)+1:end),:) ;


training_labels = Y(idx(1:round(P*m)),:) ; 
testing_labels = Y(idx(round(P*m)+1:end),:) ;

training_dcds = dcds_array(idx(1:round(P*m)),:) ; 
testing_dcds = dcds_array(idx(round(P*m)+1:end),:) ;

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
    
    x56 = vertcat(x56, training_vectors(i,:));
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
    
    x56 = vertcat(x56, testing_vectors(i,:));
end

X_test = horzcat(x1,x2,x3,x4,x56,x7);
Y_test = X_test * trained_weights;



dcds_test = [];
for i = 1:length(testing_labels)
    tmp_term = 0;
    
    % Compute the derivative for the 4 hidden neurons
    for J = 1:4
        coeff = trained_weights(J) / 2;
        denominator = mahalanobis(testing_vectors(i,:),GMModel.mu(J,:),GMModel.Sigma);
        numerator = 2 * GMModel.Sigma(1) * (testing_vectors(i,1) - GMModel.mu(J,1)) + ...
            2 * GMModel.Sigma(2) * (testing_vectors(i,2) - GMModel.mu(J,2));

        tmp_term = tmp_term + coeff * numerator / denominator; 
    end
    % Add the derivative of the linear term
    tmp_term = tmp_term + trained_weights(5);
    
    % Store the computed derivative for that sample
    dcds_test = vertcat(dcds_test, tmp_term);
end

% Plot figure 5b
% dc/dS, T-t, C/X
figure();
F = scatteredInterpolant(X_test(:,5),X_test(:,6),dcds_test);
F.Method = 'linear';

min_sx = min(X_test(:,5));
min_ttm = min(X_test(:,6));
max_sx = max(X_test(:,5));
max_ttm = max(X_test(:,6));
proj_sx = linspace(min_sx, max_sx, 200);
proj_ttm = linspace(min_ttm, max_ttm, 200);
[PROJ_SX, PROJ_TTM] = ndgrid(proj_sx, proj_ttm);
PROJ_CX = F(PROJ_SX, PROJ_TTM);


mesh(PROJ_SX, PROJ_TTM, PROJ_CX);
colormap winter;
hold on;
ylabel('T-t');
xlabel('S/K');
zlabel('Delta');
hold off;



% Plot figure 5d
% dc/dS, T-t, C/X
figure();
F = scatteredInterpolant(X_test(:,5),X_test(:,6),dcds_test - testing_dcds);
F.Method = 'linear';

min_sx = min(X_test(:,5));
min_ttm = min(X_test(:,6));
max_sx = max(X_test(:,5));
max_ttm = max(X_test(:,6));
proj_sx = linspace(min_sx, max_sx, 200);
proj_ttm = linspace(min_ttm, max_ttm, 200);
[PROJ_SX, PROJ_TTM] = ndgrid(proj_sx, proj_ttm);
PROJ_CX = F(PROJ_SX, PROJ_TTM);


mesh(PROJ_SX, PROJ_TTM, PROJ_CX);
colormap winter;
hold on;
ylabel('T-t');
xlabel('S/K');
zlabel('Delta Error');
hold off;







% 
% % Setup the design matrix for testing split
% x1 = [];
% x2 = [];
% x3 = [];
% x4 = [];
% x56 = [];
% x7 = ones(length(testing_labels),1);
% for i = 1:length(testing_labels)
%     x1 = vertcat(x1, mahalanobis(testing_vectors(i,:),GMModel.mu(1,:),GMModel.Sigma));
%     x2 = vertcat(x2, mahalanobis(testing_vectors(i,:),GMModel.mu(2,:),GMModel.Sigma));
%     x3 = vertcat(x3, mahalanobis(testing_vectors(i,:),GMModel.mu(3,:),GMModel.Sigma));
%     x4 = vertcat(x4, mahalanobis(testing_vectors(i,:),GMModel.mu(4,:),GMModel.Sigma));
%     
%     x56 = vertcat(x56, X(i,:));
% end
% 
% X_test = horzcat(x1,x2,x3,x4,x56,x7);
% Y_test = X_test * trained_weights;
% 
% % Plot figure 5a
% % S/X, T-t, C/X
% figure();
% F = scatteredInterpolant(X_test(:,5),X_test(:,6),Y_test);
% F.Method = 'linear';
% 
% min_sx = min(X_test(:,5));
% min_ttm = min(X_test(:,6));
% max_sx = max(X_test(:,5));
% max_ttm = max(X_test(:,6));
% proj_sx = linspace(1.05, max_sx, 200);
% proj_ttm = linspace(min_ttm, max_ttm, 200);
% [PROJ_SX, PROJ_TTM] = ndgrid(proj_sx, proj_ttm);
% PROJ_CX = F(PROJ_SX, PROJ_TTM);
% 
% 
% mesh(PROJ_SX, PROJ_TTM, PROJ_CX);
% colormap winter;
% hold on;
% ylabel('T-t');
% xlabel('S/K');
% zlabel('C/K');
% hold off;
% 
% % Plot figure 5c
% figure();
% F = scatteredInterpolant(X_test(:,5),X_test(:,6),Y_test - testing_labels);
% F.Method = 'linear';
% 
% min_sx = min(X_test(:,5));
% min_ttm = min(X_test(:,6));
% max_sx = max(X_test(:,5));
% max_ttm = max(X_test(:,6));
% proj_sx = linspace(1.05, max_sx, 200);
% proj_ttm = linspace(min_ttm, max_ttm, 200);
% [PROJ_SX, PROJ_TTM] = ndgrid(proj_sx, proj_ttm);
% PROJ_CX = F(PROJ_SX, PROJ_TTM);
% 
% 
% mesh(PROJ_SX, PROJ_TTM, PROJ_CX);
% colormap winter;
% hold on;
% ylabel('T-t');
% xlabel('S/K');
% zlabel('C/K Error');
% hold off;


