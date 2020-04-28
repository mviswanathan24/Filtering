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

mean_vals = zeros(3,5);
P_vals = zeros(3,3,5);

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

mean_vals(:,1) = x;
P_vals(:,:,1) = P;

norm_error = zeros(3,1);
net_error = zeros(3,1);

x_act = x;
actual = zeros(3,6);
actual(:,1) = x_act;

%ensemble points
Xen = zeros(3,100);
%data matrix 
D = zeros(1,100);

for j=1:100
    Xen(:,j) = mvnrnd(x,P,1);
end

for t=1:50
for i=1:5
    x1 = x_act(1);
    x2 = x_act(2);
    x3 = x_act(3);
    fval = double(subs(f))*dt;
    x_act = x_act + fval + G*mvnrnd(0,Q,1);
    actual(:,i+1) = x_act;
    
    
    %dynamics of ensemble points
    for j=1:100
        x1 = Xen(1,j);
        x2 = Xen(2,j);
        x3 = Xen(3,j);
        fval = double(subs(f))*dt;
        Xen(:,j) = Xen(:,j) + fval;
    end
    
    %expectation of ensemble points
    EXen = mean(Xen')';
    
    A = Xen - EXen;
    C = A*A'/(100-1);
    
    %recalculation of D matrix
    z_val = H*x_act;
    for j=1:100
        D(:,j) = z_val + mvnrnd(0,100,1);
    end
    
    %kalman gain calculation
    K = C*H'*inv(H*C*H'+R);
    Xen = Xen + K*(D-H*Xen);
    
    %update mean and covariance
    mean_vals(:,i+1) = mean(Xen')';
    A = Xen - EXen;
    
    norm_error = norm_error + norm(A);
    
    C = (A*A')/(100-1);
    P_vals(:,:,i+1) = C;
end
end

rmse = vpa(norm_error/50);


x_axis = [1:6];
figure(1)
errorbar(x_axis, mean_vals(1,:),[P_vals(1,1,1),P_vals(1,1,2),P_vals(1,1,3),P_vals(1,1,4),P_vals(1,1,5),P_vals(1,1,6)])

figure(2)
errorbar(x_axis, mean_vals(2,:),[P_vals(2,2,1),P_vals(2,2,2),P_vals(2,2,3),P_vals(2,2,4),P_vals(2,2,5),P_vals(1,1,6)])

figure(3)
errorbar(x_axis, mean_vals(3,:),[P_vals(3,3,1),P_vals(3,3,2),P_vals(3,3,3),P_vals(3,3,4),P_vals(3,3,5),P_vals(1,1,6)])


