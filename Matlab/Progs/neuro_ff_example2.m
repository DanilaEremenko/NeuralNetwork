function neuro_ff_example2
global class_data;
global classType;
global data_type_names axis_rect data_type;
global data_type_per_class C;
global colors markertypes;

close all; % �������� ���� �������� ����

colors = {[1 0 0], [0 1 0], [ 0 0 1], [1 1 0], [1 0 1], [0 1 1], 1/255*[255 165 0], 1/255*[47 79 79], 1/255*[188 143 143], 1/255*[139 69 19], 1/255*[0 191 255] };  % ����� ��� ������� ������
markertypes = { 'x', 'o', '*', 'diamond','x', 'o', '*', 'diamond', 'x', 'o', '*', 'diamond' };    % ���� �������� ��� ������� ������

% ��������� �������
class_data = GenerateDataWrap('SetDataVariants');
use_example_distributions = 1; % ���� 1, �� ������������ ������� - ���������, ���� 0, �� �������� ����������� ������� (��. ������� GenerateSamples2)
if use_example_distributions
    data_type = 10;  % ��� ������������� ������ �� 1 �� 11
    % 1 - Normal, 2 - GMM,  3- Uniform, 4 - Norm+GMM+Uniform, 5 - Conc. Circles, 6 - Hor. Stripes, 7 - Ver. Stripes, 8 - Squares, 9 - Circle+Lines, 10 - Spiral, 11 - Mozaic
    Ntrain = 1000;   % ����� ��������� ������� (���������� �������� ��� ������ ������)
    Ntest = 1000;   % ����� �������� ������� (���������� �������� ��� ������ ������)
    is_2_class = 0;     % ���� 1, �� ��� ������������� ������������ ������ 2 ������, ���������� n_class_2
    n_class_2 = [2 3]; % ����� ������ ������������ ��� ������������� � 2 ��������
    
    if is_2_class
        [PtestC, TtestC] = GenerateDataWrap( 'GenerateData', data_type, Ntest*ones(1, C(data_type)), n_class_2);
        [PtrainC, TtrainC] = GenerateDataWrap( 'GenerateData', data_type, Ntrain*ones(1, C(data_type)), n_class_2);
        C(data_type) = 2;
    else
        [PtestC, TtestC] = GenerateDataWrap( 'GenerateData', data_type, Ntest*ones(1, C(data_type)), 1:C(data_type));
        [PtrainC, TtrainC] = GenerateDataWrap( 'GenerateData', data_type, Ntrain*ones(1, C(data_type)), 1:C(data_type));
    end;
else
    NtrainAll = 1000;   % ����� ��������� �������
    NtestAll = 2000;   % ����� �������� �������
    [PtrainC, TtrainC, PtestC, TtestC] = GenerateSamples2(NtrainAll, NtestAll);
end;

% ��������, ��������, ������������� �� 
NN_hidden_neurons = [100 ];       % ���������� �������� � ������� �����
type_train_func = 6;    % ������� �������� (1-traingd, 2-traingda, 3-traingdm, 4-traingdx, 5-trainrp, 6-traincgf, 7-traincgb, 8-traincgp, 9-trainscg, 10-trainlm, 11-trainbfg, 12-trainoss, 13-trainbr)
type_perform_fcn = 5;   % �������������� ������� (1-mae, 2-mse, 3-sae, 4-sse, 5-crossentropy)
NN_arch_type = 1;   % ����������� (1 - ������� �� ������� ���������������, 2 - ��������� �� ��)
NN_out_type = 5;    % �� ��������� ���� (1 - purelin, 2-tansig, 3-logsig, 4-satlin, 5 - softmax), softmax ���������� ������������ � ��������� � crossentropy

% �������� �� ������� ���������������
net = create_neural_network(NN_arch_type, NN_out_type, NN_hidden_neurons);
net = init(net);        % ������������� ������� ������������� � ��������
net = SetTrainParam(net, type_train_func, type_perform_fcn);
net = train_neural_network(net, PtrainC, TtrainC);
ytrain = sim_neural_network(net, PtrainC);  % ����� �� �� ��������� �������
ytest = sim_neural_network(net, PtestC);    % ����� �� �� �������� �������
% ����������� ������ �� ������ �� �������� ��������� ��������� �������
[~,ytrain_c] = max(ytrain); [~,ytest_c] = max(ytest);

% ���������� ��������
h_figures = GenerateFigures;
% ������ ���������� ������������� �������� �� ��������� �������
plot_type1(h_figures{1}, PtrainC, TtrainC)
% ������ ���������� ������������� �������� �� �������� �������
plot_type1(h_figures{2}, PtestC, TtestC)
% ������ ����������� ������������� ���������� �������� �� �������� �������
plot_type1(h_figures{3}, PtestC, ytest_c)
% ������ � ���������� ��������
plot_type2(h_figures{4}, net);
% ������ ������
[e_mean, e1, e2, c] = calc_errors(ytest_c, TtestC)
% ������ ������� ������
plot_confusion_matrix(h_figures{5}, TtestC, ytest_c);
SetContextMenus(h_figures);
end
%% ������ ������
function [e_mean, e1, e2, c] = calc_errors(ytest_c, TtestC)
e_mean = sum(sum(ytest_c ~= TtestC)) / length(TtestC)
c = confusionmat(TtestC',ytest_c')
c1 = c - diag(diag(c));
e1 = sum(c1)./sum(c)
e2 = sum(c1')./sum(c')
end
%% ��������� �������
function [PtrainC, TtrainC, PtestC, TtestC] = GenerateSamples2(NtrainAll, NtestAll)
global data_type axis_rect C;
% ��������� ���������� ����������� �������
% PtestC - ������� ������� �������� ��� �������� ������� �����������
% ����� �������� �������� * ����� ��������� (2)
% PtrainC - ������� ������� �������� ��� ��������� ������� �����������
% ����� ��������� �������� * ����� ��������� (2)
% TtestC - ������� �������� �������� �������� ��� �������� ������� �����������
% 1 * ����� �������� ��������
% TtrainC - ������� �������� �������� �������� ��� ��������� ������� �����������
% 1 * ����� ��������� ��������
% TtrainC(1,i)=j ��������, ��� i-� ��������� ������ ��������� � ������ j
% ��������� ������
% data_type - ����� ����� ������� (�.�. ������ 11)
% C(datatype) - ����� �������
% axis_rect(datatype) - ���������� ������� ������������� ������� �������� �
% ������� [xmin xmax ymin ymax]

data_type = 12;
C(data_type) = 3;
axis_rect{data_type} = [0 1 0 1];

PtrainC = rand(NtrainAll,2);
PtestC = rand(NtestAll,2);
TtrainC = zeros(1,NtrainAll);
TtestC = zeros(1,NtestAll);
% ��� ������� ��� ������, � ������� x<0.3, ������� � 1 ������,
TtrainC(1,PtrainC(:,1)<.3) = 1;
TtestC(1,PtestC(:,1)<.3,1) = 1;
% ��� ������, � ������� 0.3<=x<0.6, ������� � 2 ������
TtrainC(1,PtrainC(:,1)>=.3 & PtrainC(:,1)<.6) = 2;
TtestC(1,PtestC(:,1)>=.3 & PtestC(:,1)<.6) = 2;
% ��� ������, � ������� x>=0.6, ������� � 3 ������
TtrainC(1,PtrainC(:,1)>=.6) = 3;
TtestC(1,PtestC(:,1)>=.6) = 3;
end
%% NN - �������
% �������� ��
function net = create_neural_network(NN_arch_type, NN_out_type, NN_hidden_neurons)
net = [];
if NN_arch_type == 1
    % ff
     net = feedforwardnet(NN_hidden_neurons);
elseif NN_arch_type == 2
     net = cascadeforwardnet(NN_hidden_neurons);
    %cascade
end;
if NN_out_type == 1
    % sigmoid
    net.layers{length(net.layers)}.transferFcn = 'purelin';
elseif NN_out_type == 2
    % sigmoid
    net.layers{length(net.layers)}.transferFcn = 'logsig';
elseif NN_out_type == 3
    % sigmoid
    net.layers{length(net.layers)}.transferFcn = 'tansig';
elseif NN_out_type == 4
    % sigmoid
    net.layers{length(net.layers)}.transferFcn = 'satlin';
elseif NN_out_type==5
    net.layers{net.numLayers}.transferFcn = 'softmax';
%    net.plotFcns = {'plotperform','plottrainstate','ploterrhist',...
%    'plotconfusion','plotroc'};
    
    % softmax
end;
end
% �������� ��
function net = train_neural_network(net, dataTrain, groupsTrain)
C = max(groupsTrain);
NS_out = zeros(C,length(groupsTrain));

for i = 1:C
    NS_out(i,:) = (groupsTrain == i);
end;
net = train(net, dataTrain', NS_out);
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
trainParam.max_fail = 6;            % ������������ ����� ���� ��� ������� ��������

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
% ������������� ��
function y = sim_neural_network(net, dataAll)
y = sim(net, dataAll');
end
%% ���������� ��������
function h_figures = GenerateFigures
posX=100;
posY=400;
figH=300;   %height of GUI (pixels)
figW=400;  %width of GUI (pixels)

% �������� �����
h1 = figure;
set(h1, 'OuterPosition',[posX posY figW figH ], 'Name','Train Sample, Points, True Data','NumberTitle','off', 'Toolbar', 'none', 'Menubar', 'none');
h2 = figure;
set(h2, 'OuterPosition',[posX posY-figH figW figH ], 'Name','Test Sample 1, Points, True Data','NumberTitle','off', 'Toolbar', 'none', 'Menubar', 'none');
h3 = figure;
set(h3, 'OuterPosition',[posX+figW posY figW figH ], 'Name','Test Sample 1, Points, Neural network answer','NumberTitle','off', 'Toolbar', 'none', 'Menubar', 'none');
h4 = figure;
set(h4, 'OuterPosition',[posX+figW posY-figH figW figH ], 'Name','Test Sample 2, Surface, Neural network answer','NumberTitle','off', 'Toolbar', 'none', 'Menubar', 'none');
h5 = figure;
set(h5, 'OuterPosition',[posX+2*figW posY-figH figW*2 figH*2 ], 'Name','Confusion Matrix','NumberTitle','off', 'Toolbar', 'none', 'Menubar', 'none');

h_figures = {h1, h2, h3, h4, h5};
end
function plot_type1(h_figure, P, T)
global C data_type axis_rect colors markertypes;
set(0, 'CurrentFigure', h_figure);
hold on; grid on;
for c = 1:C(data_type)
    plot(P(T==c,1), P(T==c,2), 'linestyle','none', 'color',colors{c}, 'marker', markertypes{c});
end;
xlabel('x_{1}'); ylabel('x_{2}');
axis(axis_rect{data_type});
end
function plot_type2(h_figure, net)
global C data_type axis_rect colors markertypes;

set(0, 'CurrentFigure', h_figure);
hold on;
deltaT2 = [0.05 0.05];
x1 = axis_rect{data_type}(1):deltaT2(1):axis_rect{data_type}(2);
x2 = axis_rect{data_type}(3):deltaT2(2):axis_rect{data_type}(4);
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
Y = sim_neural_network(net, X);    % ����� �� �� �������� �������
[~,Z] = max(Y);
Z = reshape(Z,length(x2), length(x1));

cmap_cur = [];
for i = 1:C(data_type)
    cmap_cur = [cmap_cur; colors{i}];
end;
i_scatter = 1:size(cmap_cur,1);
for i = i_scatter
    scatter(0,0,1,cmap_cur(i,:),'filled');
end;
colormap(cmap_cur);
surf(X1,X2,Z, 'edgecolor','none', 'CData', Z+1 );
view(2)
axis(axis_rect{data_type});
grid on;
str_leg = {};
for i = 1:C(data_type)
    str_leg = [str_leg ['Class ' num2str(i)]];
end;
legend(str_leg);
end
function plot_confusion_matrix(h_figure, TtestC, ytest_c)
set(0, 'CurrentFigure', h_figure);
plotconfusion(ind2vec(TtestC),ind2vec(ytest_c));
end
%% �������� ����������, ����
% �������, �������������� ����������� ������� � ����� ������ 
function SetContextMenus(h_figures)
% �������� ����������� ����
copy_menu = {};
for i = 1:length(h_figures)
    copy_menu{i} = uicontextmenu;
    set(copy_menu{i}, 'Parent', h_figures{i});
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
newFig = figure;%('visible','off');
for j = 1:length(haxes)
    copyobj(haxes(j),newFig);
end;
set(newFig, 'colormap',cmap);

filename = datestr(now, 'mmmm dd, yyyy HH-MM-SS.FFF');
switch type
    case 1
        print(newFig,'-dmeta');
    case 2
        print(newFig,'-dbitmap');
    case 3
        print(newFig,'-dpdf');
    case 4
        print(newFig, [filename '.emf'],'-dmeta');
    case 5
        print(newFig, [filename '.jpg'],'-djpeg');
    case 6
        print(newFig, [filename  '.pdf'],'-dpdf');
end;
delete(newFig);    
end