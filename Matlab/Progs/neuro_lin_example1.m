function neuro_lin_example1
close all;
P=0:.01:1; w = 2;
T = sin(w*P);
maxlr = 0.10*maxlinlr(P,'bias');
delta_epochs = 10;
n_delta = 30;
net = newlin(P,1,0, maxlr);
net.iw{1} = .2;
net.b{1} = 0;
net.trainparam.epochs = delta_epochs;
w = zeros(n_delta +1,2); w(1,:) = [net.iw{1} net.b{1}];
for i = 1: n_delta
net = train(net,P,T);
w(i+1,1) = net.iw{1};
w(i+1,2) = net.b{1};
end;
subplot(2,1,1);
w_range = 0:0.1:1;  b_range = 0:0.1:1;
ES = errsurf(P,T,w_range,b_range,'purelin'); 
contour(w_range, b_range, ES); hold on; grid on;
plot(w(:,1), w(:,2), '-ro');
xlabel('w_1'); ylabel('b');
subplot(2,1,2);
plot(P,T, '-ro');
hold on; grid on;
Y = sim(net,P);
plot(P,Y, '-bx');
legend({'Desire', 'Real'});
xlabel('x'); ylabel('y');
