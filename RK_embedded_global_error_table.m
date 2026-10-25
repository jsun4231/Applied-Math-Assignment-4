
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

Bogacki.A = [0,0,0,0; 1/2,0,0,0; 0,3/4,0,0; 2/9,1/3,4/9,0];
Bogacki.B = [2/9,1/3,4/9,0; 7/24,1/4,1/3,1/8];
Bogacki.C = [0,1/2,3/4,1];

BT_high = Bogacki;
BT_high.B = Bogacki.B(1,:)';
BT_low = Bogacki;
BT_low.B = Bogacki.B(2,:)';

method_names = {'Higher-order method (RK3)'; 'Lower-order method (RK2)'; 'Difference |XB1-XB2|'};
errors = zeros(length(h_refs),3);
step_sizes = zeros(length(h_refs),1);
p = zeros(3,1);

for j = 1:length(h_refs)
    [~,X_high,h_avg] = explicit_RK_fixed_step_integration(rate_func,tspan,V0,h_refs(j),BT_high);
    [~,X_low] = explicit_RK_fixed_step_integration(rate_func,tspan,V0,h_refs(j),BT_low);

    errors(j,1) = norm(X_high(end,:)' - V_true);
    errors(j,2) = norm(X_low(end,:)' - V_true);
    errors(j,3) = norm(X_high(end,:)' - X_low(end,:)');
    step_sizes(j) = h_avg;
end

for i = 1:3
    fit = polyfit(log(step_sizes),log(errors(:,i)),1);
    p(i) = fit(1);
end

results = table(method_names,p,'VariableNames',{'Method','p (-)'});

fprintf('Table 1. Global truncation error scaling with step size for the Bogacki-Shampine embedded RK 3(2) method (E ~ h^p)\n\n');
disp(results)

fileID = fopen('RK_embedded_global_error_table.html','w');
fprintf(fileID,'<html><body style="font-family:Arial;font-size:12pt">\n');
fprintf(fileID,'<p><b>Table 1. Global truncation error scaling with step size for the Bogacki-Shampine embedded RK 3(2) method (E ~ h^p)</b></p>\n');
fprintf(fileID,'<table border="1" cellspacing="0" cellpadding="8">\n');
fprintf(fileID,'<tr><th>Method</th><th>Scaling exponent, p (-)</th></tr>\n');

for i = 1:3
    fprintf(fileID,'<tr><td>%s</td><td>%.3f</td></tr>\n',method_names{i},p(i));
end

fprintf(fileID,'</table></body></html>\n');
fclose(fileID);

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
