function neuro_ff_example1

close all; % �������� ���� �������� ����

tstart = 1;     % ��������� �������� ���������
tfinish = 10;   % �������� �������� ���������

Ntrain = 300;   % ����� ��������� �������
Ntest = 1000;   % ����� �������� �������

% ������������ ��������� ������� � ����������� �����
Ptrain = tstart + (tfinish-tstart) * [0:1/Ntrain:1];
% ������������ ��������� ������� ���������� ���������� �� ���������
Ptest = tstart + (tfinish-tstart)*sort(rand(1,Ntest));

% ������� - ����� 3-� ��������
w = [5 7 8];
phi = [0 1 2];
k = [1 1 1];
my_func = @(x)(k(1)*sin(w(1)*x+phi(1)) + k(2)*sin(w(2)*x+phi(2)) + k(3)*sin(w(3)*x+phi(3)));

% ���������� �������� (target) �������� ������������ �������
Ttrain = my_func(Ptrain);
Ttest = my_func(Ptest);

% �������� �� ������� ���������������

%-------------------------------------------------------------------------%
%---------------------������� ������� � ���������� ��������---------------%
%-------------------------------------------------------------------------%

NN_hidden_neurons = [ 50 ];       % ���������� �������� � ������� �����
type_train_func = 6;    % ������� �������� (1-traingd, 2-traingda, 3-traingdm, 4-traingdx, 5-trainrp, 6-traincgf, 7-traincgb, 8-traincgp, 9-trainscg, 10-trainlm, 11-trainbfg, 12-trainoss, 13-trainbr)
type_perform_fcn = 4;   % �������������� ������� (1-mae, 2-mse, 3-sae, 4-sse, 5-crossentropy)
NN_arch_type = 2;   % ����������� (1 - ������� �� ������� ���������������, 2 - ��������� �� ��)

net = create_neural_network(NN_arch_type, NN_hidden_neurons);
net = init(net);        % ������������� ������� ������������� � ��������
net = SetTrainParam(net, type_train_func, type_perform_fcn);
net.trainParam

net = train(net, Ptrain, Ttrain);   % �������� ��
ytrain = sim(net, Ptrain);          % ����� �� �� ��������� �������
ytest = sim(net, Ptest);            % ����� �� �� �������� �������

% ������ ������
% ������-�������������� ��������
etrain_std = std(ytrain - Ttrain)
etest_std = std(ytest - Ttest)
% ������������ �������� �� ������
etrain_max = max(abs(ytrain - Ttrain))
etest_max = max(abs((ytest - Ttest)))

h_figures = GenerateFigures;

subplot(2,1,1);
plot(Ptrain, Ttrain, '-ro' );
hold on; grid on;
plot(Ptrain, ytrain, '-bx');
legend({'Real', 'NN'});
title(sprintf('Training sample, e_{std} = %g, e_{max} = %g', etrain_std, etrain_max));

subplot(2,1,2);
plot(Ptest, Ttest, '-ro' );
hold on; grid on;
plot(Ptest, ytest, '-bx');
legend({'Real', 'NN'});
title(sprintf('Test sample, e_{std} = %g, e_{max} = %g', etest_std, etest_max));
SetContextMenus(h_figures);
end

%% NN - �������
% �������� ��
function net = create_neural_network(NN_arch_type, NN_hidden_neurons)
net = [];
if NN_arch_type == 1
    % ff
     net = feedforwardnet(NN_hidden_neurons);
elseif NN_arch_type == 2
     net = cascadeforwardnet(NN_hidden_neurons);
    %cascade
end;
end
% ������� ���������� ��
function net = SetTrainParam(net, type_train_func, type_perform_fcn)
trainf_fcns = {'traingd','traingda', 'traingdm', 'traingdx', 'trainrp'... % 1-5
    'traincgf', 'traincgb', 'traincgp', 'trainscg',... % 6-9
    'trainlm', 'trainbfg', 'trainoss', 'trainbr' };   % 10-13
perform_fcns = {'mae', 'mse', 'sae', 'sse', 'crossentropy'};
net.trainfcn = trainf_fcns{type_train_func};        % ������� ��������
net.performfcn = perform_fcns{type_perform_fcn};    % ������� ���������� ������


if type_perform_fcn == 5
    net.performParam.regularization = 0.1;
    net.performParam.normalization = 'none';
    net.outputs{length(net.layers)}.processParams{2}.ymin = 0;
end;

trainParam = net.trainParam;

trainParam.epochs = 1000;           % ������������ �������� ����� ���� ��������
trainParam.time = Inf;              % ������������ ����� ��������
trainParam.goal = 0;                % ������� �������� ������
trainParam.min_grad = 1e-05;        % �������� ��������� ��� ��������
trainParam.max_fail = 16;            % ������������ ����� ���� ��� ������� ��������

% ��������� ������������ ��������
trainParam.showWindow = true;       % ���������� ���� ��� ���
trainParam.showCommandLine = false; % �������� � ��������� ������ ��� ���
trainParam.show = 25;               % ������� ���������� - ����� ������� ����

switch type_train_func
    case 1
        % traingd - ����������� �����
        trainParam.lr = 0.01;               % !�������� ��������
    case 2
        % traingda - ����������� ����� c ����������
        trainParam.lr = 0.01;               % !�������� �������� (�����������)
        trainParam.lr_inc = 1.05;           % !����������� ���������� �������� ��������
        trainParam.lr_dec = 0.7;            % !����������� ���������� �������� ��������
        trainParam.max_perf_inc  = 1.04;    % !���������� ����������� ��������� ������ 
                                                % ��� ��� ���������� �������� ����������� � lr_dec ���, ������������ �� ����������, � ��������� ������ ������������ ����������
                                                % ���� ������� ������ ������ ����������, �� �������� ������������� � lr_inc ���
    case 3
        % traingm - ����������� ����� c ����������
        trainParam.lr = 0.01;               % !�������� ��������
        trainParam.mc = 0.9;                % !������ ������� (�� 0 �� 1), ��� �� ������ ��� ����� ������� ��������� �������������
                                                % ��� mc=0 traingdm ��������� � traingd
    case 4
        % traingx - ����������� ����� c ���������� � ��������
        trainParam.lr = 0.01;               % !�������� �������� (�����������)
        trainParam.mc = 0.9;                % !������ �������
        trainParam.lr_inc = 1.05;           % !����������� ���������� �������� ��������
        trainParam.lr_dec = 0.7;            % !����������� ���������� �������� ��������
        trainParam.max_perf_inc  = 1.04;    % !���������� ����������� ��������� ������ 
    case 5
        % trainrp
        trainParam.lr = 0.01;               % !�������� �������� (�����������)
        % ��������� ��������� (��� ������ �������� delta)
        trainParam.delt_inc = 1.2;          % Increment to weight change
        trainParam.delt_dec = 0.5;          % Decrement to weight change
        trainParam.delta0 = 0.07;           % Initial weight change
        trainParam.deltamax = 50.0;         % Maximum weight change
    case {6,7,8, 11, 12}
        % traincgf, traincgp, traincgb, trainbfg, trainoss
        if ~isempty(find(type_train_func == [6 7 8], 1))
            trainParam.searchFcn = 'srchcha';   % !������� ����������� ��������� ������ (srchbac, srchbre, srchgol, srchhyb)
        else
            trainParam.searchFcn = 'srchbac';   % !������� ����������� ��������� ������ (srchbac, srchbre, srchgol, srchhyb)
        end
        % ��������� ������� ����������� ������
        trainParam.scale_tol = 20;         % Divide into delta to determine tolerance for linear search.
        trainParam.alpha = 0.001;           % Scale factor that determines sufficient reduction in perf
        trainParam.beta = 0.1;              % Scale factor that determines sufficiently large step size
        trainParam.delta = 0.01;            % Initial step size in interval location step
        trainParam.gama = 0.1;              % Parameter to avoid small reductions in performance, usually set to 0.1 (see srch_cha)
        trainParam.low_lim = 0.1;           % Lower limit on change in step size
        trainParam.up_lim = 0.5             % Upper limit on change in step size
        trainParam.max_step = 100;           % Maximum step length
        trainParam.min_step = 1.0e-6;        % Minimum step length
        trainParam.bmax = 26;               % Maximum step size
        if type_train_func == 11
            trainParam.batch_frag = 0;          % In case of multiple batches, they are considered independent. Any nonzero value implies a fragmented batch, so the final layer's conditions of a previous trained epoch are used as initial conditions for the next epoch.
        end;
    case 9
        % trainscgf
        trainParam.sigma = 5e-5;            % ��������� ����� ��� ������������� ������ �����������
        trainParam.lambda = 5e-7;           % �������� ��� ������������� ��� ������ �������������� ������� �����
    %-------------------------------------------------------%
    %-------������ ���������� �������-----------------------%
    %-------------------------------------------------------%
    case {10, 13}
        % trainlm, trainbr
        % ��������� ��������� (��� ������ �������� mu)
        trainParam.mu = 0.001;              % Initial mu
        trainParam.mu_dec = 0.1;            % mu decrease factor
        trainParam.mu_inc = 10;             % mu increase factor
        trainParam.mu_max = 1e10;           % Maximum mu
end;
net.trainParam = trainParam;
end
%% ���������� ��������
function h_figures = GenerateFigures
posX=100;
posY=400;
figH=600;   %height of GUI (pixels)
figW=800;  %width of GUI (pixels)

% �������� �����
h1 = figure;
set(h1, 'OuterPosition',[posX posY figW figH ], 'Name','Train Sample, Points, True Data','NumberTitle','off', 'Toolbar', 'none', 'Menubar', 'none');
h_figures = {h1};
end
%% �������� ����������, ����
% �������, �������������� ����������� ������� � ����� ������ 
function SetContextMenus(h_figures)
% �������� ����������� ����
copy_menu = {};
for i = 1:length(h_figures)
    copy_menu{i} = uicontextmenu(h_figures{i});
    topmenu1 = uimenu('Parent',copy_menu{i},'Label','Copy');
    topmenu2 = uimenu('Parent',copy_menu{i},'Label','Save');
    childmenu1 = uimenu('Parent', topmenu1, 'Label', 'As meta-file', 'Callback', @(x,y)(CopySaveMenuItem_Callback(x,y, 1)));
    childmenu2 = uimenu('Parent', topmenu1, 'Label', 'As bitmap', 'Callback', @(x,y)(CopySaveMenuItem_Callback(x,y, 2)) );
%    childmenu3 = uimenu('Parent', topmenu1, 'Label', 'As pdf', 'Callback',@(x,y)(CopySaveMenuItem_Callback(x,y, 3)));
    childmenu4 = uimenu('Parent', topmenu2, 'Label', 'As emf', 'Callback', @(x,y)(CopySaveMenuItem_Callback(x,y, 4)));
    childmenu5 = uimenu('Parent', topmenu2, 'Label', 'As jpeg', 'Callback',@(x,y)(CopySaveMenuItem_Callback(x,y, 5)) );
    childmenu6 = uimenu('Parent', topmenu2, 'Label', 'As pdf', 'Callback', @(x,y)(CopySaveMenuItem_Callback(x,y, 6)));
end;
% �������������� ����������� ���� � ��������� � ����� (axes)
for i = 1:length(h_figures)
    haxes = findall(h_figures{i}, 'Type', 'Axes');
    for j = 1:length(haxes)
    set(haxes(j), 'UIcontextmenu', copy_menu{i});
    hlines = findall(haxes(j));
    % Attach the context menu to each line
    for i_line = 1:length(hlines)
        set(hlines(i_line),'uicontextmenu',copy_menu{i})
    end
    end;
end;
end
% ����������� ������� � �����/���������� � ����
function CopySaveMenuItem_Callback(x,y,type)
haxes = findall(gcf, 'Type', 'Axes');
cmap = get(gcf, 'colormap');
newFig = figure('visible','off');
for j = 1:length(haxes)
    copyobj(haxes(j),newFig);
end;
set(newFig, 'colormap',cmap);

filename = datestr(datetime('now'), 'mmmm dd, yyyy HH-MM-SS.FFF');
switch type
    case 1
        print(newFig,'-clipboard','-dmeta');
    case 2
        print(newFig,'-clipboard','-dbitmap');
    case 3
        print(newFig,'-clipboard','-dpdf');
    case 4
        print(newFig, [filename '.emf'],'-dmeta');
    case 5
        print(newFig, [filename '.jpg'],'-djpeg');
    case 6
        print(newFig, [filename  '.pdf'],'-dpdf');
end;
delete(newFig);    
end