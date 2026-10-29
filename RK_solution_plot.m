clear
clc

orbit_params.m_sun = 1;
orbit_params.m_planet = 1;
orbit_params.G = 40;

V0 = [8; 0; 0; 1.5];
h_ref = 0.01;

% orbital period from Kepler's third law, simulate three periods
mu = orbit_params.G*orbit_params.m_sun;
E = 0.5*norm(V0(3:4))^2 - mu/norm(V0(1:2));
a_orbit = -mu/(2*E);
T = 2*pi*sqrt(a_orbit^3/mu);
tspan = [0, 3*T];

rate_func = @(t,V) gravity_rate_func(t,V,orbit_params);

ExplicitMidpoint.A = [0 0; 1/2 0];
ExplicitMidpoint.B = [0; 1];
ExplicitMidpoint.C = [0; 1/2];

Bogacki.A = [0,0,0,0; 1/2,0,0,0; 0,3/4,0,0; 2/9,1/3,4/9,0];
Bogacki.B = [2/9; 1/3; 4/9; 0];
Bogacki.C = [0; 1/2; 3/4; 1];

RK4.A = [0 0 0 0; 1/2 0 0 0; 0 1/2 0 0; 0 0 1 0];
RK4.B = [1/6; 1/3; 1/3; 1/6];
RK4.C = [0; 1/2; 1/2; 1];

methods = {ExplicitMidpoint, Bogacki, RK4};
method_names = {'Explicit Midpoint'; 'Bogacki-Shampine RK3'; 'Runge-Kutta 4'};
markers = {'o','s','^'};
colors = lines(3);

% exact solution on a fine grid
t_exact = linspace(tspan(1),tspan(2),2000)';
V_exact = compute_planetary_motion(t_exact,V0,orbit_params);

figure('Color','w','Position',[100 100 1100 800])
tl = tiledlayout(2,1,'TileSpacing','compact');
comp_labels = {'$x(t)$','$y(t)$'};
ax = gobjects(2,1);

for c = 1:2
    ax(c) = nexttile;
    hold on
    plot(t_exact,V_exact(:,c),'k-','LineWidth',2,'DisplayName','Exact solution')
end

mark_every = 60;
for i = 1:length(methods)
    [t_list,X_list] = explicit_RK_fixed_step_integration(rate_func,tspan,V0,h_ref,methods{i});
    V_ref = compute_planetary_motion(t_list,V0,orbit_params);
    fprintf('%s: max |x err| = %.3e, max |y err| = %.3e\n',method_names{i}, ...
        max(abs(X_list(:,1) - V_ref(:,1))),max(abs(X_list(:,2) - V_ref(:,2))))

    idx = (1 + (i-1)*mark_every/3):mark_every:length(t_list);
    for c = 1:2
        plot(ax(c),t_list(idx),X_list(idx,c),markers{i},'Color',colors(i,:), ...
            'MarkerSize',9,'LineWidth',1.5,'DisplayName',method_names{i})
    end
end

for c = 1:2
    set(ax(c),'FontSize',16,'TickLabelInterpreter','latex')
    grid(ax(c),'on')
    box(ax(c),'on')
    xlim(ax(c),tspan)
    ylabel(ax(c),comp_labels{c},'Interpreter','latex','FontSize',20)
end
set(ax(1),'XTickLabel',[])

title(tl,sprintf('Exact vs. Explicit RK Solutions for Planetary Motion ($h=%.2f$)',h_ref), ...
    'Interpreter','latex','FontSize',22)
xlabel(tl,'Time, $t$','Interpreter','latex','FontSize',20)

lgd = legend(ax(1),'Orientation','horizontal','Interpreter','latex','FontSize',14);
lgd.Layout.Tile = 'south';

exportgraphics(gcf,'RK_solution_plot.png','Resolution',300)


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
