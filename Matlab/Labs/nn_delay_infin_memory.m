function nn_delay_infin_memory

close all;

% Task params
type = 1;
do_prepare = 1;
% Common NN params
delays = 0:1:10;
% Type 1 NN params
lr = .001;
% Type 2 NN params
n_hidden = 15;

mem_deep_u = 0;

% Первый порядок
mem_deep_x = 1;
z1 = 0.9;
d = 1;
f_infin_mem = @(u,x,n)(d*u(n)+z1*x(n-1));

% Второй порядок
mem_deep_x = 2;
z1 = 0.4 + .6i;
z2 = 0.4 - .6i;
d = 1;
f_infin_mem = @(u,x,n)(d*u(n)+(z1+z2)*x(n-1) - z1*z2*x(n-2));


amin = -10;
amax = 10;
tmin = 5;
tmax = 10;
t0=0; % start-time
t1=200; % end time
n=30; % Average number of time cycles in the interval (t0, t1)
dt = 0.1;
da = 0.01;
noise_amp = 0;

[t_train,u_train] =  generate_RTS(t0, t1, dt, da, n, tmin, tmax, amin, amax, noise_amp);
[t_test,u_test] =  generate_RTS(t0, t1, dt, da, n, tmin, tmax, amin, amax, noise_amp);


x_train = zeros(size(t_train));
x_test = zeros(size(t_test));
for i = 1:length(t_train)
    mem_deep = max([mem_deep_u mem_deep_x]);
    if i <= mem_deep
        x_train(i) = f_infin_mem([zeros(1, mem_deep) u_train], [zeros(1, mem_deep) x_train], i + mem_deep);
        x_test(i) = f_infin_mem([zeros(1, mem_deep) u_train], [zeros(1, mem_deep) x_test], i + mem_deep);
    else
        x_train(i) = f_infin_mem(u_train, x_train, i);
        x_test(i) = f_infin_mem(u_test, x_test, i);
    end;
end;

u_train_c = num2cell(u_train);
x_train_c = num2cell(x_train);
u_test_c = num2cell(u_test);
x_test_c = num2cell(x_test);

switch type
    case 1
        net = newlin(u_train_c, x_train_c, delays, lr)
        %net = linearlayer(delays, lr)
        net.trainfcn = 'trains';
        Xi = {};
    case 2
        net = timedelaynet(delays, n_hidden);
        net.trainfcn = 'trainlm';
end;
net = init(net);
%gensim(net)

if do_prepare
    [Xs,Xi,Ai,Ts] = preparets(net,u_train_c,x_train_c);
    net = train(net,Xs,Ts,Xi,Ai);
    [Xs,Xi,Ai,Ts] = preparets(net,u_test_c,x_test_c);
    y_nn = net(Xs,Xi,Ai);
else
    net = train(net, u_train_c, x_train_c);
    y_nn = sim(net, u_test_c);
end;



err_nn = cell2mat(y_nn) - x_test(length(Xi)+1:end);

net.iw{1}
net.b{1}

subplot(3,1,1);
plot(t_test,u_test);
ylabel( 'u' );
title( 'Input Sequence' );
grid on;

subplot(3,1,2);
plot(t_test,x_test);
grid on;
hold on;
plot(t_test(length(Xi)+1:end),cell2mat(y_nn), 'color', 'r');
ylabel( 'x' );
title( 'Output Sequence' );

subplot(3,1,3);
plot(t_test(length(Xi)+1:end),err_nn, 'color', 'b');
grid on;
ylabel( 'e' );
title( 'Error' );
%plot(x(length(Xi)+1:end),cell2mat(y_nn), 'color', 'r');
%y_nn;


function [t,x] = generate_RTS(t0, t1, dt, da, n, tmin, tmax, amin, amax, noise_amp)

% Create time vector
t_rand = tmin + (tmax-tmin) * rand(n,1);
t_rand = round(t_rand/dt) * dt;
% 
t = t0:dt:t1;
amps = amin + (amax-amin) * rand(n,1);
amps = round(amps/da) * da;
x = zeros(size(t));

x(t<t_rand(1)) = amps(1);

for i = 2:n
    x(t<sum(t_rand(1:i)) & t >= sum(t_rand(1:i-1))) = amps(i);
end;

x = x + rand(size(x))*noise_amp;