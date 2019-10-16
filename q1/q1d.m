% Load in the specific dataset
% TTM (expressed in years)
T = (277 - 135) / 365;

% Strike Price
K = 7300;

% Risk free interest rate
r = 0.01267;

% Stock price on that date
S0 = 7617.7;

% Estimated volatility
sigma = 0.11412013507029374;

% BS output from python
BS_price = 433.4412656706718;

% Compute the Binomial Lattice output for different N
disp('Price     deltaT');
lattice_prices = [];
deltaT_lst = [];

% N - No of time steps
for N=4:5
    [price, lattice] = LatticeEurCall(S0,K,r,T,sigma,N);
    deltaT = T/N;
    lattice_prices = [lattice_prices, price];
    deltaT_lst = [deltaT_lst, deltaT];
end

% Find the absolute difference between the 2 methods
abs_diff = abs(lattice_prices - BS_price);

% Plot the absolute diferrence between the 2 method against dt.
plot(deltaT_lst, abs_diff);
ylabel('Absolute Difference');
xlabel('Step Time, \deltat');
grid();


% The actual price of the market
actual_price = 402;


% for tau=1:N
%     for i= (tau+1):2:(2*N+1-tau)
%         disp([i, tau]);
%     end
% end
% 
% 
% for i=1:2:2*N+1
%     disp(i);
% end