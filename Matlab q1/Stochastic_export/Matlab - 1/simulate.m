%% File Info.

%{

    simulate.m
    ----------
    This code simulates the model.

%}

%% Simulate class.

classdef simulate
    methods(Static)
        %% Simulate the model. 
        
        function sim = lc(par,sol)            
            %% Set up.
            
           [agrid, apol, cpol, npol, nt_T] = deal(par.agrid, sol.a, sol.c, sol.n, sol.nt_T);
           [TT, NN, T, tr, Gt, kappa, ygrid, pmat] = deal(par.TT, par.NN, par.T, par.tr, par.Gt, par.kappa, par.ygrid, par.pmat);
           [ysim, asim, tsim, csim, usim, nsim] = deal(nan(TT,NN));
           yr = nan(NN,1);

            
            %% Begin simulation.
            
            rng(par.seed);
            pmat0 = pmat^100; % Stationary distribution
            cmat = cumsum(pmat,2); % CDF matrix

            y0_ind = randsample(par.ylen,NN,true,pmat0(1,:))';
            a0_ind = randsample(par.alen,NN,true)';
            t0_ind = ones(T,NN)'; % Initial ages
            yr = nan(NN,1); % Pension buffer

            for i = 1:NN
                n = npol(a0_ind(i),t0_ind(i),y0_ind(i));
                nsim (1,i) = npol(a0_ind(i),t0_ind(i),y0_ind(i));

                if t0_ind(i) >= tr
                    nt_temp = nt_T(y0_ind(i), a0_ind(i));
                    yr(i) = ygrid(y0_ind(i));
                    ysim(1,i) = kappa.* yr(i) * Gt(tr-1)*nt_temp;
                else
                    ysim(1,i) = Gt(1) * ygrid(y0_ind(i)) * n;
                end

                tsim(1,i) = t0_ind(i);
                csim(1,i) = cpol(a0_ind(i), t0_ind(i), y0_ind(i));
                asim(1,i) = apol(a0_ind(i), t0_ind(i), y0_ind(i));

                if t0_ind(i) == tr - 1
                    yr(i) = ygrid(y0_ind(i));
                elseif t0_ind(i) < tr - 1
                    y1_ind = find(rand <= cmat(y0_ind(i),:));
                    y0_ind(i) = y1_ind(1);
                end
            end

            usim(1,:) = model.utility(csim(1,:),nsim(1,:), par);

            %% Simulate next periods
            
            for j = 2:TT
                for i = 1:NN
                    age = tsim(j-1,i) + 1;

                    if age <= T
                        tsim(j,i) = age;
                        at_ind = find(asim(j-1,i)==agrid);
                        n = npol(at_ind,age,y0_ind(i));
                        nsim(j,i) = npol(at_ind,age,y0_ind(i));
                        if age >= tr
                            nt_temp = nt_T(y0_ind(i), at_ind);
                            ysim(j,i) = kappa.* yr(i) * Gt(tr-1)*nt_temp;
                        else
                            ysim(j,i) = Gt(age)*ygrid(y0_ind(i))*n;
                        end

                        
                        csim(j,i) = cpol(at_ind, age, y0_ind(i));
                        asim(j,i) = apol(at_ind, age, y0_ind(i));
                        usim(j,i) = model.utility(csim(j,i),n, par);

                        if age == tr - 1
                            yr(i) = ygrid(y0_ind(i));
                        elseif age < tr - 1
                            y1_ind = find(rand <= cmat(y0_ind(i),:));
                            y0_ind(i) = y1_ind(1);
                        end
                    end
                end
            end

            %% Output structure
            sim = struct();
            sim.ysim = ysim; sim.asim = asim;
            sim.tsim = tsim; sim.csim = csim;
            sim.usim = usim; sim.nsim = nsim;
        end
    end
end