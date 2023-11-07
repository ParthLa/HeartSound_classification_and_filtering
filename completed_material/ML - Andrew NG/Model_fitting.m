theta_0=0;
theta_1=0;
function [val1] = model_fit(alpha,m,x,y)
val1=0;
val2=0;
while(val2<=val1)
    delta_sum = 0;
    delta_sum_x = 0;
    for i=1:m
        h_theta = theta_0 + theta_1*x(i);
        delta_sum = delta_sum + (h_theta - y(i));
        delta_sum_x = delta_sum_x + (h_theta - y(i))*x(i);
    end
    theta_0 = theta_0 - alpha*delta_sum/m;
    theta_1 = theta_1 - alpha*delta_sum_x/m;
    J_sum=0;
    for i=1:m
        h_theta= theta_0 + theta_1*x(i);
        J_sum = J_sum +  (h_theta - y(i))^2;
    end
    J_sum = J_sum /(2*m);
    val1=val2;
    val2= J_sum;
end
end