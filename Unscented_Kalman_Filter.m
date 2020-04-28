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

f1 = x2;
f2 = d - g;
f3 = 0;

f = [f1;f2;f3];
f_new = [x1+f1*dt; x2+f2*dt; x3]; %x1+ f1*dt or just f1? mostly the latter
f_jacobian = jacobian(f_new,[x1, x2, x3]);

%based on medium article and Cyrill Stachniss
n = 3;
K = 10;
alpha = 0.5;
beta = 2;
lambda = alpha*alpha*(n+K) - n;

%mean weights
w0 = lambda/(n+lambda);
w1 = 1/(2*(n+lambda));
w2 = 1/(2*(n+lambda));
w3 = 1/(2*(n+lambda));
w4 = 1/(2*(n+lambda));
w5 = 1/(2*(n+lambda));
w6 = 1/(2*(n+lambda));

wts = [w0 w1 w2 w3 w4 w5 w6];

%covariance weights
wts_C = [w0+(1-alpha*alpha)+beta w1 w2 w3 w4 w5 w6];

sigs_pred = zeros(3,7);
sigs_pred2 = zeros(3,7);
Z_expected = zeros(1,7);

%to store system dynamics vals
x_act = x;
actual = zeros(3,6);
actual(:,1) = x_act;

% mean_vals(:,1) = x;
% P_vals(:,:,1) = P;

for i=1:5
    mean_vals(:,i) = x;
    P_vals(:,:,i) = P;
    x1 = x_act(1);
    x2 = x_act(2);
    x3 = x_act(3);
    f0 = double(subs(f_new));
    x_act = f0 + G*mvnrnd(0,Q,1);
    actual(:,i+1) = x_act;   %check later
    
    %sigma points 
    sig0 = x;
    sg = sqrt(n+lambda)*chol(P)';
    sig1 = x + sg(:,1);
    sig2 = x + sg(:,2);
    sig3 = x + sg(:,3);
    sig4 = x - sg(:,1);
    sig5 = x - sg(:,2);
    sig6 = x - sg(:,3);
    
    sigs = [sig0 sig1 sig2 sig3 sig4 sig5 sig6];
    
    %prediction step
    %mean prediction
    x_bar = 0;
    f_pred = zeros(3,7);
    for j=1:7
        x1 = sigs(1,j);
        x2 = sigs(2,j);
        x3 = sigs(3,j);
        fval = double(subs(f_new))*dt;
        sigs_pred(:,j) = fval;
        x_bar = x_bar + wts(j)*sigs_pred(:,j);
    end
    
    %covariance prediction
    P_bar = zeros(3,3);
    for j=1:7
        P_bar = P_bar + wts_C(j)*(sigs_pred(:,j)-x_bar)*(sigs_pred(:,j)-x_bar)'; %check later
    end
    P_bar = P_bar + G*Q*G';
    
    %update
    %measurement
    %sigma points using predicted mean and covariance values
    sg2 = sqrt(n+lambda)*chol(P_bar)';
    sigs_pred2(:,1) = x_bar;
    sigs_pred2(:,2) = x_bar + sg2(:,1);
    sigs_pred2(:,3) = x_bar + sg2(:,2);
    sigs_pred2(:,4) = x_bar + sg2(:,3);
    sigs_pred2(:,5) = x_bar - sg2(:,1);
    sigs_pred2(:,6) = x_bar - sg2(:,2);
    sigs_pred2(:,7) = x_bar - sg2(:,3);
    
    %expected measurement
    Z_expected = H*sigs_pred2; 
    
    %weighting the measurements, refer to as zhat
    zhat = 0;
    for j=1:7
        zhat = zhat + wts(j)*Z_expected(j);
    end
    
    %noise correlation matrix S
    S = 0;
    for j=1:7
        S = S + wts(j)*(Z_expected(j) - zhat)*(Z_expected(j) - zhat)';
    end
    S = S + R;
    
    %cross correlation terms
    T = 0;
    for j=1:7
        T = T + wts(j)*(sigs_pred(:,j) - x_bar)*(Z_expected(j) - zhat)';
    end
    
    %kalman gain
    Kg = T*inv(S);
    
    %updated mean and covariance
    z_act = H*x_act + mvnrnd(0,R);
    x = x_bar + Kg*(z_act - zhat);
    P = P_bar - Kg*S*Kg';   %cyril says something else though apparently
    
    mean_vals(:,i+1) = x;
    P_vals(:,:,i+1) = P;
    
end

x_axis=1:6;

figure(1)
errorbar(x_axis,mean_vals(1,:),[P_vals(1,1,1),P_vals(1,1,2),P_vals(1,1,3),P_vals(1,1,4),P_vals(1,1,5),P_vals(1,1,6)])

figure(2)
errorbar(x_axis,mean_vals(2,:),[P_vals(2,2,1),P_vals(2,2,2),P_vals(2,2,3),P_vals(2,2,4),P_vals(2,2,5),P_vals(2,2,6)])

figure(3)
errorbar(x_axis,mean_vals(3,:),[P_vals(3,3,1),P_vals(3,3,2),P_vals(3,3,3),P_vals(3,3,4),P_vals(3,3,5),P_vals(3,3,6)])



