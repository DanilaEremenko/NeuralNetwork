function nn_delay
close all;

% Task params
type = 1;
do_prepare = 1;

% function params
w = 3;
f_predict = @(x) (sin(w*x) + cos(2*w*x));
deltaX = .02;
% Predict params
dX_predict = .5;
% Common NN params
delays = 0:5;
% Type 1 NN params
lr = .1;
% Type 2 NN params
n_hidden = 5;


x = 0:deltaX:10;
y1 = f_predict(x);
y2 = f_predict(x+dX_predict);
subplot(2,1,1);
plot(x,y1, 'color', 'k');
hold on; grid on;
plot(x,y2, 'color', 'r');

y1_c = num2cell(y1);
y2_c = num2cell(y2);

switch type
    case 1
        net = newlin(y1, y2, delays, lr)
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
    [Xs,Xi,Ai,Ts] = preparets(net,y1_c,y2_c);
    net = train(net,Xs,Ts,Xi,Ai);
    y_nn = net(Xs,Xi,Ai);
else
    net = train(net, y1_c, y2_c);
    y_nn = sim(net, y1_c);
end;

net.iw{1}
net.b{1}

subplot(2,1,2);
plot(x,y2, 'color', 'b');
hold on; grid on;
plot(x(length(Xi)+1:end),cell2mat(y_nn), 'color', 'r');
y_nn;