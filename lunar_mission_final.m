function lunar_mission_final

clc; clear; close all;

use_simulated_camera = true;    
use_simulated_initial_estimate = true;

e            = 0.20;     
alt_peri     = 100e3;    
meas_sigma   = 300;      

% user can request position and velocity at t = ? hours
query_times_hours = [1.0, 2.0, 4.5];
query_times = query_times_hours * 3600;

% moon gravity and moon radius
mu    = 4.9048695e12;
Rmoon = 1737.4e3;

% orbit parameters
rp = Rmoon + alt_peri;
a  = rp / (1 - e);
ra = a * (1 + e);
T_orbit = 2*pi*sqrt(a^3/mu);

vp = sqrt(mu * (2/rp - 1/a));

% true initial state
state_true = [rp; 0; 0; 0; vp; 0];

% initial state estimate uncertainty
sig_x  = 200;   
sig_y  = 200;   
sig_z  = 150;  
sig_vx = 0.4;  
sig_vy = 0.4;   
sig_vz = 0.2;   

state_true0 = state_true;

% time vector
tf = 2*T_orbit;      
N = 1000;
tvec = linspace(0, tf, N);

MC = 100;   

ex_all = zeros(N, MC);
ey_all = zeros(N, MC);
ez_all = zeros(N, MC);

% store one run for plotting/tables
Xtrue_plot = [];
Xest_plot  = [];
tex = [];

for mc = 1:MC

    state_true = state_true0;

    if use_simulated_initial_estimate
        state_est = state_true + [ ...
            sig_x*randn;
            sig_y*randn;
            sig_z*randn;
            sig_vx*randn;
            sig_vy*randn;
            sig_vz*randn ];
    else
        load("initial_camera_state.mat","state_est");
    end

    % EKF covariance matrices
    P = diag([sig_x^2 sig_y^2 sig_z^2 sig_vx^2 sig_vy^2 sig_vz^2]);
    R = (meas_sigma^2) * eye(3);
    Q = zeros(6);

    Xtrue = zeros(N,6);
    Xest  = zeros(N,6);

    Xtrue(1,:) = state_true.';
    Xest(1,:)  = state_est.';

    % EKF integration loop
    for k = 2:N

        dt = tvec(k) - tvec(k-1);

        % propagate true state
        initialF = eye(6);
        state_true_ext = [state_true; initialF(:)];
        [~, traj_true] = ode45(@(t,s) twoBody_STM(t,s,mu), [0 dt], state_true_ext);
        state_true = traj_true(end,1:6)';

        % camera measurement
        if use_simulated_camera
            z_meas = state_true(1:3) + meas_sigma*randn(3,1);
        else
            z_meas = camera_data(:,k);
        end

        % EKF prediction
        initialF = eye(6);
        state_est_ext = [state_est; initialF(:)];
        [~, traj_est] = ode45(@(t,s) twoBody_STM(t,s,mu), [0 dt], state_est_ext);
        state_est = traj_est(end,1:6)';

        F = reshape(traj_est(end,7:42), 6, 6);
        P = F*P*F.' + Q;

        % EKF update
        H = eye(3,6);
        z_hat = state_est(1:3);
        y = z_meas - z_hat;

        S = H*P*H.' + R;
        K = P*H.'/S;

        state_est = state_est + K*y;
        P = (eye(6) - K*H)*P;

        Xtrue(k,:) = state_true.';
        Xest(k,:)  = state_est.';
    end

    ex_all(:,mc) = Xest(:,1) - Xtrue(:,1);
    ey_all(:,mc) = Xest(:,2) - Xtrue(:,2);
    ez_all(:,mc) = Xest(:,3) - Xtrue(:,3);

    if mc == 1
        Xtrue_plot = Xtrue;
        Xest_plot  = Xest;

        fprintf("\n================= QUERY RESULTS TABLE =================\n");

        QueryResults = table();
        for i = 1:length(query_times)

            tq = query_times(i);

            x_true_interp = interp1(tvec', Xtrue, tq);
            x_est_interp  = interp1(tvec', Xest,  tq);

            newRow = table( ...
                query_times_hours(i), ...
                x_true_interp(1), x_true_interp(2), x_true_interp(3), ...
                x_true_interp(4), x_true_interp(5), x_true_interp(6), ...
                x_est_interp(1),  x_est_interp(2),  x_est_interp(3), ...
                x_est_interp(4),  x_est_interp(5),  x_est_interp(6), ...
                'VariableNames', { ...
                'Time_hr', ...
                'True_x','True_y','True_z','True_vx','True_vy','True_vz', ...
                'EKF_x','EKF_y','EKF_z','EKF_vx','EKF_vy','EKF_vz'});

            QueryResults = [QueryResults; newRow];
        end

        disp(QueryResults);

        FinalTable = table( ...
            Xtrue(end,1), Xtrue(end,2), Xtrue(end,3), Xtrue(end,4), Xtrue(end,5), Xtrue(end,6), ...
            Xest(end,1),  Xest(end,2),  Xest(end,3),  Xest(end,4),  Xest(end,5),  Xest(end,6), ...
            'VariableNames', { ...
            'True_x','True_y','True_z','True_vx','True_vy','True_vz', ...
            'EKF_x','EKF_y','EKF_z','EKF_vx','EKF_vy','EKF_vz'});

        fprintf("\n================= FINAL STATE TABLE =================\n");
        disp(FinalTable);

        % moon graphics
        if exist("moon.jpg","file")
            tex = imread("moon.jpg");
        else
            url = "https://www.solarsystemscope.com/textures/download/2k_moon.jpg";
            websave("moon.jpg", url);
            tex = imread("moon.jpg");
        end
        tex = flipud(tex);
    end
end

% Figure 1: Lunar orbit trajectory
figure(1); clf;
set(gcf,'Color','w');
set(gcf,'Position',[100 100 850 700]);

plot3(Xtrue_plot(:,1), Xtrue_plot(:,2), Xtrue_plot(:,3), ...
    'b-', 'LineWidth', 1.8); 
hold on;

plot3(Xest_plot(:,1), Xest_plot(:,2), Xest_plot(:,3), ...
    'r--', 'LineWidth', 1.5);

% draw moon with texture
[xx,yy,zz] = sphere(150);
hs = surf(Rmoon*xx, Rmoon*yy, Rmoon*zz, ...
    'FaceColor','texturemap', ...
    'EdgeColor','none', ...
    'CData', tex);

uistack(hs,'bottom');

xlabel('X [m]','FontName','Times New Roman','FontSize',12);
ylabel('Y [m]','FontName','Times New Roman','FontSize',12);
zlabel('Z [m]','FontName','Times New Roman','FontSize',12);

title('Lunar Orbit Trajectory and EKF Estimated Trajectory', ...
    'FontName','Times New Roman', ...
    'FontSize',13, ...
    'FontWeight','bold');

legend('True Trajectory','EKF Estimated Trajectory','Moon', ...
    'Location','best', ...
    'FontName','Times New Roman', ...
    'FontSize',11);

grid on;
axis equal;
set(gca,'FontName','Times New Roman','FontSize',11,'LineWidth',1.0);
view(35,25);

% Figure 2: State estimation errors
figure(2); clf;
set(gcf,'Color','w');
set(gcf,'Position',[100 100 800 750]);

sig_x_plot = std(ex_all,0,2);
sig_y_plot = std(ey_all,0,2);
sig_z_plot = std(ez_all,0,2);

mean_x = mean(ex_all,2);
mean_y = mean(ey_all,2);
mean_z = mean(ez_all,2);

t_orbits = tvec / T_orbit;

lw_sigma = 1.5;
lw_mean  = 1.7;

tl = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
title(tl,'Spacecraft Position Estimation Errors Using EKF', ...
    'FontName','Times New Roman', ...
    'FontSize',13, ...
    'FontWeight','bold');

% X error
ax1 = nexttile;
p1 = plot(t_orbits,  3*sig_x_plot,'b-','LineWidth',lw_sigma); hold on;
p2 = plot(t_orbits, -3*sig_x_plot,'r-','LineWidth',lw_sigma);
p3 = plot(t_orbits, mean_x,'k--','LineWidth',lw_mean);
ylabel('X Error [m]','FontName','Times New Roman','FontSize',12);
grid on;
set(gca,'FontName','Times New Roman','FontSize',11,'LineWidth',1.0);
xlim([0 2]);

% Y error
ax2 = nexttile;
plot(t_orbits,  3*sig_y_plot,'b-','LineWidth',lw_sigma); hold on;
plot(t_orbits, -3*sig_y_plot,'r-','LineWidth',lw_sigma);
plot(t_orbits, mean_y,'k--','LineWidth',lw_mean);
ylabel('Y Error [m]','FontName','Times New Roman','FontSize',12);
grid on;
set(gca,'FontName','Times New Roman','FontSize',11,'LineWidth',1.0);
xlim([0 2]);

% Z error
ax3 = nexttile;
plot(t_orbits,  3*sig_z_plot,'b-','LineWidth',lw_sigma); hold on;
plot(t_orbits, -3*sig_z_plot,'r-','LineWidth',lw_sigma);
plot(t_orbits, mean_z,'k--','LineWidth',lw_mean);
ylabel('Z Error [m]','FontName','Times New Roman','FontSize',12);
xlabel('Time / Orbit Period','FontName','Times New Roman','FontSize',12);
grid on;
set(gca,'FontName','Times New Roman','FontSize',11,'LineWidth',1.0);
xlim([0 2]);

lgd = legend([p1 p2 p3], {'+3\sigma Bound','-3\sigma Bound','Mean Error'}, ...
    'Orientation','horizontal', ...
    'Location','northoutside');
lgd.Layout.Tile = 'north';
lgd.Box = 'off';
lgd.FontName = 'Times New Roman';
lgd.FontSize = 11;

set(ax1,'Position',[0.13 0.69 0.80 0.20]);
set(ax2,'Position',[0.13 0.41 0.80 0.20]);
set(ax3,'Position',[0.13 0.13 0.80 0.20]);

sgtitle('Spacecraft Position Estimation Errors Using EKF', ...
    'FontName','Times New Roman','FontSize',12,'FontWeight','bold');

end


function dX = twoBody_STM(~, X, mu)
% Phi is 6x6 STM

x  = X(1); y  = X(2); z  = X(3);
vx = X(4); vy = X(5); vz = X(6);

Phi = reshape(X(7:42),6,6);

r  = sqrt(x^2 + y^2 + z^2);
r3 = r^3;
r5 = r^5;

ax = -mu*x/r3;
ay = -mu*y/r3;
az = -mu*z/r3;

A = zeros(6,6);
A(1:3,4:6) = eye(3);

I3  = eye(3);
rrT = [x;y;z]*[x y z];
dadr = -mu*(I3/r3 - 3*rrT/r5);

A(4:6,1:3) = dadr;

Phi_dot = A*Phi;

dX = zeros(42,1);
dX(1) = vx; dX(2) = vy; dX(3) = vz;
dX(4) = ax; dX(5) = ay; dX(6) = az;
dX(7:42) = Phi_dot(:);

end
