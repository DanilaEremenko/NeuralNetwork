function neuro_ff_example1

close all; % Закрытие всех открытых окон

tstart = 1;     % Начальное значение аргумента
tfinish = 10;   % Конечное значение аргумента

Ntrain = 300;   % Объем обучающей выборки
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

% Создание НС Прямого Распространения

%-------------------------------------------------------------------------%
%---------------------Задание функций и параметров обучения---------------%
%-------------------------------------------------------------------------%

NN_hidden_neurons = [ 50 ];       % Количество нейронов в скрытых слоях
type_train_func = 6;    % Функция обучения (1-traingd, 2-traingda, 3-traingdm, 4-traingdx, 5-trainrp, 6-traincgf, 7-traincgb, 8-traincgp, 9-trainscg, 10-trainlm, 11-trainbfg, 12-trainoss, 13-trainbr)
type_perform_fcn = 4;   % Минимизируемая функция (1-mae, 2-mse, 3-sae, 4-sse, 5-crossentropy)
NN_arch_type = 2;   % Архитектура (1 - обычная НС прямого распространения, 2 - каскадная НС ПР)

net = create_neural_network(NN_arch_type, NN_hidden_neurons);
net = init(net);        % Инициализация весовых коэффициентов и смещений
net = SetTrainParam(net, type_train_func, type_perform_fcn);
net.trainParam

net = train(net, Ptrain, Ttrain);   % Обучение НС
ytrain = sim(net, Ptrain);          % Ответ НС на обучающей выборке
ytest = sim(net, Ptest);            % Ответ НС на тестовой выборке

% Расчет ошибок
% Средне-квадратическое значение
etrain_std = std(ytrain - Ttrain)
etest_std = std(ytest - Ttest)
% Максимальное значение по модулю
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

%% NN - функции
% Создание НС
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
% Задание параметров НС
function net = SetTrainParam(net, type_train_func, type_perform_fcn)
trainf_fcns = {'traingd','traingda', 'traingdm', 'traingdx', 'trainrp'... % 1-5
    'traincgf', 'traincgb', 'traincgp', 'trainscg',... % 6-9
    'trainlm', 'trainbfg', 'trainoss', 'trainbr' };   % 10-13
perform_fcns = {'mae', 'mse', 'sae', 'sse', 'crossentropy'};
net.trainfcn = trainf_fcns{type_train_func};        % Функция обучения
net.performfcn = perform_fcns{type_perform_fcn};    % Функция вычисления ошибки


if type_perform_fcn == 5
    net.performParam.regularization = 0.1;
    net.performParam.normalization = 'none';
    net.outputs{length(net.layers)}.processParams{2}.ymin = 0;
end;

trainParam = net.trainParam;

trainParam.epochs = 1000;           % Максимальное значение числа эпох обучения
trainParam.time = Inf;              % Максимальное время обучения
trainParam.goal = 0;                % Целевое значение ошибки
trainParam.min_grad = 1e-05;        % Значение градиента для останова
trainParam.max_fail = 16;            % Максимальное число эпох для раннего останова

% Параметры визуализации обучения
trainParam.showWindow = true;       % Показывать окно или нет
trainParam.showCommandLine = false; % Выводить в командную строку или нет
trainParam.show = 25;               % Частота обновления - через сколько эпох

switch type_train_func
    case 1
        % traingd - Градиентный спуск
        trainParam.lr = 0.01;               % !Скорость обучения
    case 2
        % traingda - Градиентный спуск c адаптацией
        trainParam.lr = 0.01;               % !Скорость обучения (изначальная)
        trainParam.lr_inc = 1.05;           % !Коэффициент увеличения скорости обучения
        trainParam.lr_dec = 0.7;            % !Коэффициент уменьшения скорости обучения
        trainParam.max_perf_inc  = 1.04;    % !Допустимый коэффициент изменения ошибки 
                                                % при его превышении скорость уменьшается в lr_dec раз, коэффициенты не изменяются, в противном случае коэффициенты изменяются
                                                % Если текущая ошибка меньше предыдущей, то скорость увеличивается в lr_inc раз
    case 3
        % traingm - Градиентный спуск c адаптацией
        trainParam.lr = 0.01;               % !Скорость обучения
        trainParam.mc = 0.9;                % !Момент инерции (от 0 до 1), чем он больше тем более плавное изменение коэффициентов
                                                % При mc=0 traingdm переходит в traingd
    case 4
        % traingx - Градиентный спуск c адаптацией и моментом
        trainParam.lr = 0.01;               % !Скорость обучения (изначальная)
        trainParam.mc = 0.9;                % !Момент инерции
        trainParam.lr_inc = 1.05;           % !Коэффициент увеличения скорости обучения
        trainParam.lr_dec = 0.7;            % !Коэффициент уменьшения скорости обучения
        trainParam.max_perf_inc  = 1.04;    % !Допустимый коэффициент изменения ошибки 
    case 5
        % trainrp
        trainParam.lr = 0.01;               % !Скорость обучения (изначальная)
        % Параметры алгоритма (для поиска значения delta)
        trainParam.delt_inc = 1.2;          % Increment to weight change
        trainParam.delt_dec = 0.5;          % Decrement to weight change
        trainParam.delta0 = 0.07;           % Initial weight change
        trainParam.deltamax = 50.0;         % Maximum weight change
    case {6,7,8, 11, 12}
        % traincgf, traincgp, traincgb, trainbfg, trainoss
        if ~isempty(find(type_train_func == [6 7 8], 1))
            trainParam.searchFcn = 'srchcha';   % !Функция одномерного линейного поиска (srchbac, srchbre, srchgol, srchhyb)
        else
            trainParam.searchFcn = 'srchbac';   % !Функция одномерного линейного поиска (srchbac, srchbre, srchgol, srchhyb)
        end
        % Параметры функции одномерного поиска
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
        trainParam.sigma = 5e-5;            % Изменение весов для аппроксимации второй производной
        trainParam.lambda = 5e-7;           % Параметр для регуляризации при плохой обусловенности матрицы Гессе
    %-------------------------------------------------------%
    %-------Методы переменной метрики-----------------------%
    %-------------------------------------------------------%
    case {10, 13}
        % trainlm, trainbr
        % Параметры алгоритма (для поиска значения mu)
        trainParam.mu = 0.001;              % Initial mu
        trainParam.mu_dec = 0.1;            % mu decrease factor
        trainParam.mu_inc = 10;             % mu increase factor
        trainParam.mu_max = 1e10;           % Maximum mu
end;
net.trainParam = trainParam;
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