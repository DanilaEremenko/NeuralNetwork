function neuro_rbf_example1

close all; % �������� ���� �������� ����

tstart = 1;     % ��������� �������� ���������
tfinish = 10;   % �������� �������� ���������

Ntrain = 100;   % ����� ��������� �������
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

%-------------------------------------------------------------------------%
%---------------------������� ������� � ���������� ��������---------------%
%-------------------------------------------------------------------------%

type_nn_rbf = 1; % 1 - ������ ���-��, 2 - ������������ ���-��, 3 - ��� c ���������� ����������

spread = 1e-1;  % ������ ���-�������
goal_error = 1e-2;  % ������� �������� ������ � newrb
delta_neuron = 10; % �� ������� �������� ��������� ��� ������ � newrb

switch type_nn_rbf
    case 1
        net = newrbe(Ptrain, Ttrain, spread);   % �������� ��
        net.layers{1}.size
    case 2
        net = newrb(Ptrain, Ttrain, goal_error, spread, length(Ttrain), delta_neuron);   % �������� ��
        net.layers{1}.size
    case 3
        net = newgrnn(Ptrain, Ttrain, spread);   % �������� ��
        net.layers{1}.size
end;
        
ytrain = sim(net, Ptrain);          % ����� �� �� ��������� �������
ytest = sim(net, Ptest);            % ����� �� �� �������� �������

% ������ ������
% ������-�������������� ��������
etrain_std = std(ytrain - Ttrain)
etest_std = std(ytest - Ttest)
% ������������ �������� �� ������
etrain_max = max(abs(ytrain - Ttrain))
etest_max = max(abs((ytest - Ttest)))


h_figures = GenerateFigures
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