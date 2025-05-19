%% File Info.

%{
    solve.m
    -------
    This code solves the model.
%}

%% Solve class.

classdef solve
    methods(Static)
        %% Solve the model using BI. 
        
        function sol = lc(par)            
            %% Structure array for model solution.
            sol = struct();
            
            %% Model parameters, grids and functions.
            T = par.T;            % Last period of life.
            tr = par.tr;          % First year of retirement.
            gamma = par.gamma; sigma = par.sigma; nu = par.nu;            
            beta = par.beta;      % Discount factor.
            alen = par.alen;      % Grid size for a.
            agrid = par.agrid;    % Grid for a (state and choice).
            ylen = par.ylen;      % Grid size for y.
            ygrid = par.ygrid;    % Grid for y.
            pmat = par.pmat;      % Transition matrix for y.
            r = par.r;            % Real interest rate.
            kappa = par.kappa;    % Share of income as pension.
            Gt = par.Gt;          

            %% Backward induction.

            v1 = nan(alen,T,ylen); % Container for V.
            a1 = nan(alen,T,ylen); % Container for a'.
            c1 = nan(alen,T,ylen); % Container for c'.
            n1 = nan(alen,T,ylen); % Container for n.

            nt_T = zeros(ylen, alen);

            amat = repmat(agrid,1,ylen);
            ymat = repmat(ygrid,alen,1);
            
            fprintf('------------Solving from the Last Period of Life.------------\n\n')
            
            for age = 1:T % Start in the last period and iterate backward.
                
                if T-age+1 == T % Last period of life.

                    c1(:,T,:) = amat + kappa*ymat; % Consume everything.
                    a1(:,T,:) = 0.0; % Save nothing.
                    n1(:,T,:) = 0.0; % Do not work.
                    v1(:,T,:) = model.utility(c1(:,T,:),n1(:, T, :),par); % Terminal value function.
                else % All other periods.
                    for i = 1:ylen % Loop over the y-states.

                        if T-age+1 >= tr % Retirees
                            yt_base = kappa * ygrid(i) * Gt(tr-1);
                            ev = v1(:,T-age+2,i);
                        else
                            ev = squeeze(v1(:,T-age+2,:)) * pmat(i,:)';
                        end

                        for p = 1:alen
                            v_all = nan(alen,1);
                            ct_vector = nan(alen,1);
                            nt_vector = nan(alen,1);

                            for j = 1:alen
                                a_prime = agrid(j);

                                if T-age+1 == T
                                    yfunction = @(n) kappa * Gt(tr-1) * ygrid(i);
                                    cfunction = @(n) agrid(p) + Gt(T-age+1) * ygrid(i) * n - a_prime / (1+r);
                                    utilityfunction = @(n) abs(((cfunction(n).^(-sigma) * yfunction(n)) - gamma * ((1-n).^(1/nu))));
                                    [noptimal, ~] = fminbnd(utilityfunction, 0.01, 0.99);
                                    nt_T(i,p) = noptimal;
                                    nt = 0.0;
                                    yt = yt_base * noptimal;

                                elseif T-age+1 >= tr
                                    yt = yt_base * nt_T(i,p);
                                    nt = 0.0;
                                
                                else
                                    yfunction = @(n) kappa * Gt(T-age+1) * ygrid(i);
                                    cfunction = @(n) agrid(p) + Gt(T-age+1) * ygrid(i) * n - a_prime / (1+r);
                                    utilityfunction = @(n) abs(((cfunction(n).^(-sigma) * yfunction(n)) - gamma * ((1-n).^(1/nu))));
                                    [noptimal, ~] = fminbnd(utilityfunction, 0.01, 0.99);
                                    nt = noptimal;
                                    yt = Gt(T-age+1) * ygrid(i) * nt;
                                end

                                ctemp = agrid(p) + yt - a_prime / (1+r);
                                if ctemp <= 0
                                    ctemp = 0;
                                    val = -inf;
                                else
                                    val = model.utility(ctemp, nt, par) + beta * ev(j);
                                end

                                v_all(j) = val;
                                ct_vector(j) = ctemp;
                                nt_vector(j) = nt;

                            end

                            [vmax, ind] = max(v_all);
                            v1(p, T-age+1, i) = vmax;
                            c1(p, T-age+1, i) = ct_vector(ind);
                            a1(p, T-age+1, i) = agrid(ind);
                            n1(p, T-age+1, i) = nt_vector(ind);


                        end
                    end
                end

                if mod(T-age+1,5) == 0
                    fprintf('Age: %d.\n', T-age+1)
                end
            end
            
            fprintf('------------Life Cycle Problem Solved.------------\n')
            
            %% Macro variables, value, and policy functions.
            sol.c = c1; sol.a = a1; sol.v = v1; sol.n = n1; sol.nt_T = nt_T;
            
        end
    end
end
