function neuro_lab1_examples
close all;
s = RandStream('mt19937ar','Seed',0);
RandStream.setGlobalStream(s);
neuro_lab1_task1
end

function neuro_lab1_task1
global k b lr learntype;
k = 1;
b = 0.2;

x_diap = [0 1; 0 2];

N = 1000;
P = - x_diap(:,1) + x_diap(:,2) .* rand(2,N);

discrfunc = @discr_func2;
boundfunc = @bound_func2;


T = discrfunc(P);
k0 = (T==0); k1 = (T==1);
plot(P(1,k0),P(2,k0), 'xr');
hold on;
plot(P(1,k1),P(2,k1), 'xb');
xx = 0:.01:1;
plot(xx, boundfunc(xx), '-k');
axis([x_diap(1,:) x_diap(2,:)]);
title('Dataset');

    

net = newp(x_diap, 1);
lr = 1e0;
learntype = 2;

%net.inputweights{1}.learnfcn = 'learnp_custom';
%net.biases{1}.learnfcn = 'learnp_custom';
train_type = 2;

if train_type == 1
    net.trainfcn = 'trains';
    Ptrain = mat2cell(P,2,ones(1,N));
    Ttrain = mat2cell(T,1,ones(1,N));
elseif train_type == 2
    Ptrain = P;
    Ttrain = T;
    net.trainfcn = 'trainb';
end;
net.trainParam.epochs = 1e3;
net = train(net, Ptrain, Ttrain);
    
    
    y = sim(net, P);

figure;
k0 = y==0 & T == 0; k1 = y==1 & T==1;
k01 = y==0 & T == 1; k10 = y==1 & T == 0;
plot(P(1,k0),P(2,k0), 'xr');
hold on;
plot(P(1,k1),P(2,k1), 'xb');

plot(P(1,k01),P(2,k01), 'or');
plot(P(1,k10),P(2,k10), 'ob');
legend({'0-0', '1-1', '1-0', '0-1'});


xx = 0:.01:1;
plot(xx, boundfunc(xx), '-k');
axis([x_diap(1,:) x_diap(2,:)]);
title('Neural network work results');


end

function y = discr_func1(x)
global k b;
y = (k*x(1,:) + b - x(2,:) > 0);
end

function y = bound_func1(x1)
global k b;
y = k*x1 + b ;
end


function y = discr_func2(x)
global k b;
y = (k*x(1,:).^2 + b - x(2,:) > 0);

end

function y = bound_func2(x1)
global k b;
y = k*x1.^2 + b ;
end
