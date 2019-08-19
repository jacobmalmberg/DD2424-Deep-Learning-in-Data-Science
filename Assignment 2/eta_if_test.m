eta_min = 1e-5;
eta_max = 1e-1;
n_s = 800;
l=0;
cycles=3;
t= 2*l*n_s +1;
eta_t_mat = zeros(2*n_s*cycles,0);
steps_per_cycle = 2*n_s;
i=0;

while i < cycles
    
    if (t > 2*(l+1)*n_s)
        t= 2*l*n_s;
        i=i+1;
        
    end
        
    if (2*l*n_s <= t) && (t <= (2*l +1)*n_s)
        eta_t = eta_min + ((t-2*l*n_s)/n_s)*(eta_max-eta_min);
        %disp("if");
        (2*l +1)*n_s;
        eta_t_mat(i*steps_per_cycle+t) = eta_t;

    elseif ((2*l+1)*n_s <= t) && (t <= 2*(l+1)*n_s)
        eta_t = eta_max - ((t-(2*l+1)*n_s)/n_s)*(eta_max-eta_min);
        %disp("elif");
        2*(l+1)*n_s;
        eta_t_mat(i*steps_per_cycle+t) = eta_t;
        t;
    end
    t;
    
    t = t+1;
    
end
plot(eta_t_mat)
