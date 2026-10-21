clear
clc

orbit_params.m_sun = 1;
orbit_params.m_planet = 1;
orbit_params.G = 40;

V0 = [8; 0; 0; 1.5];
tspan = [0, 4];
h_refs = [0.2, 0.1, 0.05, 0.025, 0.0125];

rate_func = @(t,V) gravity_rate_func(t,V,orbit_params);
V_true = compute_planetary_motion(tspan(2),V0,orbit_params);

ForwardEuler.A = 0;
ForwardEuler.B = 1;
ForwardEuler.C = 0;

ExplicitMidpoint.A = [0 0; 1/2 0];
ExplicitMidpoint.B = [0; 1];
ExplicitMidpoint.C = [0; 1/2];

RK4.A = [0 0 0 0; 1/2 0 0 0; 0 1/2 0 0; 0 0 1 0];
RK4.B = [1/6; 1/3; 1/3; 1/6];
RK4.C = [0; 1/2; 1/2; 1];

methods = {ForwardEuler, ExplicitMidpoint, RK4};
method_names = {'Forward Euler'; 'Explicit Midpoint'; 'Runge-Kutta 4'};
p = zeros(length(methods),1);

for i = 1:length(methods)
    errors = zeros(length(h_refs),1);
    step_sizes = zeros(length(h_refs),1);

    for j = 1:length(h_refs)
        [~,X_list,h_avg] = explicit_RK_fixed_step_integration(rate_func,tspan,V0,h_refs(j),methods{i});
        errors(j) = norm(X_list(end,:)' - V_true);
        step_sizes(j) = h_avg;
    end

    fit = polyfit(log(step_sizes),log(errors),1);
    p(i) = fit(1);
end

results = table(method_names,p,'VariableNames',{'Method','p (-)'});
fprintf('Table 1. Global truncation error scaling with step size (E ~ h^p)\n\n');
disp(results)

function [t_list,X_list,h_avg,num_evals] = explicit_RK_fixed_step_integration(rate_func_in,tspan,X0,h_ref,BT_struct)
    n_steps = max(1,ceil((tspan(2) - tspan(1))/h_ref));
    t_list = linspace(tspan(1),tspan(2),n_steps + 1)';
    h_avg = (tspan(2) - tspan(1))/n_steps;
    X_list = zeros(n_steps + 1,numel(X0));
    X_list(1,:) = X0(:)';
    num_evals = 0;

    for n = 1:n_steps
        [XB,evals] = explicit_RK_step(rate_func_in,t_list(n),X_list(n,:)',h_avg,BT_struct);
        X_list(n + 1,:) = XB';
        num_evals = num_evals + evals;
    end
end

function [XB,num_evals] = explicit_RK_step(rate_func_in,t,XA,h,BT_struct)
    XA = XA(:);
    s = length(BT_struct.C);
    K = zeros(numel(XA),s);

    for i = 1:s
        X_stage = XA + h*K*BT_struct.A(i,:)';
        K(:,i) = rate_func_in(t + BT_struct.C(i)*h,X_stage);
    end

    XB = XA + h*K*BT_struct.B(:);
    num_evals = s;
end

function dVdt = gravity_rate_func(t,V,orbit_params)
    r = V(1:2);
    mu = orbit_params.G*orbit_params.m_sun;
    dVdt = [V(3:4); -mu*r/norm(r)^3];
end
