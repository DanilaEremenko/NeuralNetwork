function neuro_rbf_example1

close all; % Закрытие всех открытых окон

tstart = 1;     % Начальное значение аргумента
tfinish = 10;   % Конечное значение аргумента

Ntrain = 100;   % Объем обучающей выборки
Ntest = 1000;   % Объем тестовой выборки

% Формирование обучающей выборки с равномерным шагом
Ptrain = tstart + (tfinish-tstart) * [0:1/Ntrain:1];
% Формирование обучающей выборки случайными значениями из диапазона
Ptest = tstart + (tfinish-tstart)*sort(rand(1,Ntest));

% Функция - сумма 3-х гармоник
w = [5 7 8];
phi = [0 1 2];
k = [1 1 1];
my_func = @(x)(k(1)*sin(w(1)*x+phi(1)) + k(2)*sin(w(2)*x+phi(2)) + k(3)*sin(w(3)*x+phi(3)));

% Определяем желаемые (target) значения подстановкой функции
Ttrain = my_func(Ptrain);
Ttest = my_func(Ptest);

%-------------------------------------------------------------------------%
%---------------------Задание функций и параметров обучения---------------%
%-------------------------------------------------------------------------%

type_nn_rbf = 1; % 1 - точная РБФ-НС, 2 - приближенная РБФ-НС, 3 - РБФ c обобщенной регрессией

spread = 1e-1;  % Ширина РБФ-функции
goal_error = 1e-2;  % Целевое значение ошибки в newrb
delta_neuron = 10; % По сколько нейронов добавлять при поиске в newrb

switch type_nn_rbf
    case 1
        net = newrbe(Ptrain, Ttrain, spread);   % Обучение НС
        net.layers{1}.size
    case 2
        net = newrb(Ptrain, Ttrain, goal_error, spread, length(Ttrain), delta_neuron);   % Обучение НС
        net.layers{1}.size
    case 3
        net = newgrnn(Ptrain, Ttrain, spread);   % Обучение НС
        net.layers{1}.size
end;
        
ytrain = sim(net, Ptrain);          % Ответ НС на обучающей выборке
ytest = sim(net, Ptest);            % Ответ НС на тестовой выборке

% Расчет ошибок
% Средне-квадратическое значение
etrain_std = std(ytrain - Ttrain)
etest_std = std(ytest - Ttest)
% Максимальное значение по модулю
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
%% Построение графиков
function h_figures = GenerateFigures
posX=100;
posY=400;
figH=600;   %height of GUI (pixels)
figW=800;  %width of GUI (pixels)

% Создание фигур
h1 = figure;
set(h1, 'OuterPosition',[posX posY figW figH ], 'Name','Train Sample, Points, True Data','NumberTitle','off', 'Toolbar', 'none', 'Menubar', 'none');
h_figures = {h1};
end
%% Элементы управления, меню
% Функция, осуществляющая копирования графика в буфер обмена 
function SetContextMenus(h_figures)
% Создание контекстных меню
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
% Ассоциирование контекстных меню с графиками и осями (axes)
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
% Копирование графика в буфер/сохранение в файл
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