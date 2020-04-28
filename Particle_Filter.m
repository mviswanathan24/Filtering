m_beta = 2000;
v_beta = 2.5*10^5;
m_x1 = 10^5;
v_x1 = 500;
m_x2 = -6000;
v_x2 = 2*10^4;
m_w = 0;
s_w = 2;
kp = 22000;
g = 32.2;
rho_0 = 3.4 * 10^-3;

x = [m_x1; m_x2; m_beta];
P = diag([v_x1, v_x2, v_beta]);

%covariances

Q = s_w;
R = 200;

mean_vals = zeros(3,6);
P_vals = zeros(3,3,6);

syms x1 x2 x3 

rho = rho_0 * exp(-x1/kp);
d = rho * (x2^2)/(2*x3);
dt = 0.1;

%noise matrix
G = [0;1;0]*dt;
%measurement matrix
H = [1 0 0];

norm_error = zeros(3,1);
net_error = zeros(3,1);

f1 = x2;
f2 = d - g;
f3 = 0;

f = [f1;f2;f3];
f_new = [x1+f1*dt; x2+f2*dt; x3]; %x1+ f1*dt or just f1? mostly the latter
f_jacobian = jacobian(f_new,[x1, x2, x3]);

mean_vals(:,1) = x;
P_vals(:,:,1) = P;

x_act = x;
actual = zeros(3,6);
actual(:,1) = x_act;

%initializing with 1000 particles 
N = 1000;

X_pf = zeros(3,N);

%initializing particle distribution
for i=1:N
    X_pf(:,i) = mvnrnd(x,P,1); 
end

% initial particle distribution
figure(1)
plot(1,X_pf(1,:),'b*')
xlim([0,6])
hold on

%initial observation
Z = zeros(1,N);
Z = H*X_pf + mvnrnd(0,R,1);
figure(1)
plot(1,Z,'r*')

%assign weights to the particles
wt_P = zeros(1,N);

%dynamics propagation
% for t=1:50
for i=1:5
    x1 = x_act(1);
    x2 = x_act(2);
    x3 = x_act(3);
    fval = double(subs(f))*dt;
    x_act = x_act + fval + mvnrnd(0,Q,1);
    actual(:,i+1) = x_act;
    
    z_act = H*x_act + mvnrnd(0,R);
    
    for j=1:N
        x1 = X_pf(1,j);
        x2 = X_pf(2,j);
        x3 = X_pf(3,j);
        fval = double(subs(f))*dt;
        X_pf(:,j) = X_pf(:,j) + fval + G*mvnrnd(0,Q,1);
        
        Z(j) = H*X_pf(:,j) + mvnrnd(0,R);
        
        %weighting and normalizing the particles
        wt_P(j) = 1/(sqrt(2*pi*R))*exp(-((z_act - Z(j))^2)/(2*R)); %check later
    end
    wt_P = wt_P./sum(wt_P);
    
    figure(1)
    plot(i+1,X_pf(1,:),'m*')
    hold on 
    plot(i+1,Z,'y*')
    for j=1:N
        X_pf(:,j) = X_pf(:,find(rand <= cumsum(wt_P),1));        
    end
    
    figure(1)
    plot(i+1,X_pf(1,:),'r*')
    hold on
    plot(i+1,Z,'b*')
    hold on
    
    mean_vals(:,i+1) = mean(X_pf')';
    P_vals(:,:,i+1) = cov(X_pf');
end
% end

x_axis=[1:6];

figure(4)
errorbar(x_axis, mean_vals(1,:),[P_vals(1,1,1),P_vals(1,1,2),P_vals(1,1,3),P_vals(1,1,4),P_vals(1,1,5),P_vals(1,1,6)])
hold on
plot(mean_vals(1,:),'r*')
hold on
plot(actual(1,:),'y*')

figure(5)
errorbar(x_axis, mean_vals(2,:),[P_vals(2,2,1),P_vals(2,2,2),P_vals(2,2,3),P_vals(2,2,4),P_vals(2,2,5),P_vals(1,1,6)])
hold on
plot(mean_vals(2,:),'r*')
hold on
plot(actual(2,:),'y*')

figure(6)
errorbar(x_axis, mean_vals(3,:),[P_vals(3,3,1),P_vals(3,3,2),P_vals(3,3,3),P_vals(3,3,4),P_vals(3,3,5),P_vals(1,1,6)])
hold on
plot(mean_vals(3,:),'r*')
hold on
plot(actual(3,:),'y*')


