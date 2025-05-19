%% File Info.

%{
    model.m
    -------
    This code sets up the model.
%}

%% Model class.

classdef model
    methods(Static)
        %% Set up structure array for model parameters and set the simulation parameters.
        
        function par = setup()            
            %% Structure array for model parameters.
            
            par = struct();
           
            %% Load G_t values from CSV
            gt_data = readtable(fullfile(pwd, 'gt_by_age.csv')); 
            par.age_groups = gt_data.age; 
            par.Gt = gt_data.Gt;
            par.Gt = par.Gt / par.Gt(1);
            par.Gt = par.Gt(1:61); 

  
            %% Preferences.
            par.T = 61; % Last period of life.
            par.tr = 41; % First period of retirement.
            
            par.beta = 0.94; % Discount factor.
            par.sigma = 2.0; % CRRA.
            
            assert(par.T > par.tr,'Cannot retire after dying.\n')
            assert(par.beta > 0.0 && par.beta < 1.0,'Discount factor should be between 0 and 1.\n')
            assert(par.sigma > 0.0,'CRRA should be at least 0.\n')

            %% Prices and Income.

            par.r = 0.03; % Interest rate.
            par.kappa = 0.6; % Share of income as pension.

            par.sigma_eps = 0.07; % Std. dev of productivity shocks.
            par.rho = 0.85; % Persistence of AR(1) process.
            par.mu = 0.0; % Intercept of AR(1) process.

            assert(par.kappa >= 0.0 && par.kappa <= 1.0,'The share of income received as pension should be from 0 to 1.\n')
            assert(par.sigma_eps > 0,'The standard deviation of the shock must be positive.\n')
            assert(abs(par.rho) < 1,'The persistence must be less than 1 in absolute value so that the series is stationary.\n')

            %% Simulation parameters.

            par.seed = 2025; % Seed for simulation.
            par.TT = 61; % Number of time periods.
            par.NN = 10000; % Number of people.
        end

        %% Generate state grids.
        
        function par = gen_grids(par)
            %% Capital grid.

            par.alen = 300; % Grid size for a.
            par.amax = 30.0; % Upper bound for a.
            par.amin = 0.0; % Minimum a.
            
            assert(par.alen > 5,'Grid size for a should be positive and greater than 5.\n')
            assert(par.amax > par.amin,'Minimum a should be less than maximum value.\n')
            
            par.agrid = linspace(par.amin,par.amax,par.alen)';

            %% Discretized income process.
                  
            par.ylen = 7; % Grid size for y.
            par.m = 3; % Scaling parameter for Tauchen.
            
            assert(par.ylen > 3,'Grid size for A should be positive and greater than 3.\n')
            assert(par.m > 0,'Scaling parameter for Tauchen should be positive.\n')
            
            [ygrid,pmat] = model.tauchen(par.mu,par.rho,par.sigma_eps,par.ylen,par.m);
            par.ygrid = exp(ygrid); % Exponentiated AR(1) grid.
            par.pmat = pmat; % Transition matrix.
        end

        %% Tauchen's Method
        
        function [y,pi] = tauchen(mu,rho,sigma,N,m)
            %% Construct equally spaced grid.
            ar_mean = mu/(1-rho);
            ar_sd = sigma/sqrt(1 - rho^2);
            
            y1 = ar_mean - m*ar_sd;
            yn = ar_mean + m*ar_sd;
            
            y = linspace(y1, yn, N);
            d = y(2) - y(1);

            %% Transition matrix
            ymatk = repmat(y,N,1);
            ymatj = mu + rho * ymatk';

            pi = normcdf(ymatk, ymatj - d/2, sigma) - normcdf(ymatk, ymatj + d/2, sigma);
            pi(:,1) = normcdf(y(1), mu + rho * y - d/2, sigma);
            pi(:,N) = 1 - normcdf(y(N), mu + rho * y + d/2, sigma);
        end

        %% Utility function

        function u = utility(c,par)
            if par.sigma == 1
                u = log(c);
            else
                u = (c.^(1 - par.sigma)) / (1 - par.sigma);
            end
        end

    end
end
