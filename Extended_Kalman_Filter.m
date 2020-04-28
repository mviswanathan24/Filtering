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

syms x1 x2 x3 

rho = rho_0 * exp(-x1/kp);
d = rho * (x2^2)/(2*x3);
dt = 0.1;

f1 = x2;
f2 = d - g;
f3 = 0;

f = [f1;f2;f3];
f_new = [x1+f1*dt; x2+f2*dt; x3];
f_jacobian = jacobian(f_new,[x1, x2, x3]);

mean_vals = zeros(3,5);
P_vals = zeros(3,3,5);

norm_error = zeros(3,1);
net_error = zeros(3,1);

%noise matrix
G = [0;1;0]*dt;
%measurement matrix
H = [1 0 0];


%dynamics propagation:
%finding initial state with random num generator
 X = zeros(3,5);
 X1 = normrnd(m_x1, sqrt(v_x1));
 X2 = normrnd(m_x2, sqrt(v_x2));
 X3 = normrnd(m_beta, sqrt(v_beta));
 X(:,1) = [X1;X2;X3];
 
 for i=1:5
     x1 = X(1,i);
     x2 = X(2,i);
     x3 = X(3,i);
     X_next = X(:,i) + subs(f)*dt + G*normrnd(m_w,s_w);
     X(:,i+1) = X_next;
 end
 
for t=1:50
for i=1:5
    P = double(P);
    x = double(x);
    mean_vals(:,i) = x;
    P_vals(:,:,i) = P;
    x1 = x(1);
    x2 = x(2);
    x3 = x(3);
    %predicted mean
    x_bar = x + dt*subs(f);
    %predicted covariance
    F = subs(f_jacobian);
    P_bar = F*P*F' + G*Q*G';
    %kalman gain calculation
    Kg = P_bar*H'*inv(H*P_bar*H' + R);
    %using kalman gain to fuse the dynamics and measurement
    x = x_bar + Kg*H*(X(:,i)-x_bar); % + Kg*normrnd(0,100);
    %update covariance
    P = (eye(3)-Kg*H)*P_bar;
    x = double(x);
    P = double(P);
    
    net_error = x - x_bar;
    norm_error = norm_error + norm(net_error);

%     
%     figure(1)
%     plot(i, x(1),'r*')
%     hold on;
%     
%     figure(2)
%     plot(i, x(2),'g*')
%     hold on;
%     
%     figure(3)
%     plot(i, x(3),'b*')
%     hold on;
end

end

rmse = vpa(norm_error/50);
x_axis = 1:5;

figure(4)
errorbar(x_axis, mean_vals(1,:),[P_vals(1,1,1),P_vals(1,1,2),P_vals(1,1,3),P_vals(1,1,4),P_vals(1,1,5)])
xlim([0,6])

figure(5)
errorbar(x_axis, mean_vals(2,:),[P_vals(2,2,1),P_vals(2,2,2),P_vals(2,2,3),P_vals(2,2,4),P_vals(2,2,5)])
xlim([0,6])

figure(6)
errorbar(x_axis, mean_vals(3,:),[P_vals(3,3,1),P_vals(3,3,2),P_vals(3,3,3),P_vals(3,3,4),P_vals(3,3,5)])
xlim([0,6])
