function neuro_pnn_example
global class_data;
global classType;
global data_type_names axis_rect data_type;
global data_type_per_class C;


close all; % Закрытие всех открытых окон

colors = {[1 0 0], [0 1 0], [ 0 0 1], [1 1 0], [1 0 1], [0 1 1], 1/255*[255 165 0], 1/255*[47 79 79], 1/255*[188 143 143], 1/255*[139 69 19], 1/255*[0 191 255] };  % Цвета для каждого класса
markertypes = { 'x', 'o', '*', 'diamond','x', 'o', '*', 'diamond', 'x', 'o', '*', 'diamond' };    % Типы маркеров для каждого класса

%-----------------------------------------------%
%---------Генерация выборок---------------------%
%-----------------------------------------------%

Ntrain = 10;   % Объем обучающей выборки (количество примеров для одного класса)
Ntest = 1000;   % Объем тестовой выборки (количество примеров для одного класса)
n_class_2 = [2 3]; % Какие классы использовать в примере с ROC-кривой


class_data = GenerateDataWrap('SetDataVariants');
data_type = 9;

[PtestC, TtestC] = GenerateDataWrap( 'GenerateData', data_type, Ntest*ones(1, C(data_type)), 1:C(data_type));
[PtrainC, TtrainC] = GenerateDataWrap( 'GenerateData', data_type, Ntrain*ones(1, C(data_type)), 1:C(data_type));

[Ptest2, Ttest2] = GenerateDataWrap( 'GenerateData', data_type, Ntest*ones(1, C(data_type)), n_class_2);
[Ptrain2, Ttrain2] = GenerateDataWrap( 'GenerateData', data_type, Ntrain*ones(1, C(data_type)), n_class_2);

%-----------------------------------------------%
%----Создание, обучение, моделирование НС-------%
%-----------------------------------------------%


% Создание pnn
spread = .1;
net = newpnn(PtrainC', ind2vec(TtrainC), spread);
ytrain = sim(net, PtrainC');  % Ответ НС на обучающей выборке
ytest = sim(net, PtestC');    % Ответ НС на тестовой выборке

% Определение класса на выходе по принципу максимума выходного сигнала
ytrain_c = vec2ind(ytrain);
ytest_c = vec2ind(ytest);

%-----------------------------------------------%
%---------Построение графиков-------------------%
%-----------------------------------------------%
posX=100;
posY=400;
figH=300;   %height of GUI (pixels)
figW=400;  %width of GUI (pixels)

% Создание фигур
h1 = figure;
set(h1, 'OuterPosition',[posX posY figW figH ], 'Name','Train Sample, Points, True Data','NumberTitle','off', 'Toolbar', 'none', 'Menubar', 'none');
h2 = figure;
set(h2, 'OuterPosition',[posX posY-figH figW figH ], 'Name','Test Sample 1, Points, True Data','NumberTitle','off', 'Toolbar', 'none', 'Menubar', 'none');
h3 = figure;
set(h3, 'OuterPosition',[posX+figW posY figW figH ], 'Name','Test Sample 1, Points, Neural network answer','NumberTitle','off', 'Toolbar', 'none', 'Menubar', 'none');
h4 = figure;
set(h4, 'OuterPosition',[posX+figW posY-figH figW figH ], 'Name','Test Sample 2, Surface, Neural network answer','NumberTitle','off', 'Toolbar', 'none', 'Menubar', 'none');

h_figures = {h1, h2, h3, h4};
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

% Отрисовка графиков

% График правильной классификации примеров из обучающей выборки
set(groot, 'CurrentFigure', h1);
hold on; grid on;
for c = 1:C(data_type)
    plot(PtrainC(TtrainC==c,1), PtrainC(TtrainC==c,2), 'linestyle','none', 'color',colors{c}, 'marker', markertypes{c});
end;
axis(axis_rect{data_type});
xlabel('x_{1}'); ylabel('x_{2}');

% График правильной классификации примеров из тестовой выборки
set(groot, 'CurrentFigure', h2);
 hold on; grid on;
 for c = 1:C(data_type)
     plot(PtestC(TtestC==c,1), PtestC(TtestC==c,2), 'linestyle','none', 'color',colors{c}, 'marker', markertypes{c});
 end;
 axis(axis_rect{data_type});
 xlabel('x_{1}'); ylabel('x_{2}');

% График результатов классификации нейросетью примеров из тестовой выборки
set(groot, 'CurrentFigure', h3);
hold on; grid on;
for c = 1:C(data_type)
    plot(PtestC(ytest_c==c,1), PtestC(ytest_c==c,2), 'linestyle','none', 'color',colors{c}, 'marker', markertypes{c});
end;
xlabel('x_{1}'); ylabel('x_{2}');
axis(axis_rect{data_type});

% График с раскраской областей
set(groot, 'CurrentFigure', h4);
hold on;
deltaT2 = [0.05 0.05];
x1 = axis_rect{data_type}(1):deltaT2(1):axis_rect{data_type}(2);
x2 = axis_rect{data_type}(3):deltaT2(2):axis_rect{data_type}(4);
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
Y = sim(net, X');    % Ответ НС на тестовой выборке
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

% Ассоциирование контекстных меню с графиками и осями (axes)
for i = 1:length(h_figures)
    set(h_figures{i}.CurrentAxes, 'UIcontextmenu', copy_menu{i});
    hlines = findall(h_figures{i}.CurrentAxes,'Type','line');
    % Attach the context menu to each line
    for i_line = 1:length(hlines)
        set(hlines(i_line),'uicontextmenu',copy_menu{i})
    end
    hsurfs = findall(h_figures{i}.CurrentAxes,'Type','surface');
    % Attach the context menu to each line
    for i_surf = 1:length(hsurfs)
        set(hsurfs(i_surf),'uicontextmenu',copy_menu{i})
    end
end;


% Расчет ошибок
% Средне-квадратическое значение
e_mean = sum(sum(ytest_c ~= TtestC)) / length(TtestC)
c = confusionmat(TtestC',ytest_c')
c1 = c - diag(diag(c));
e1 = sum(c1)./sum(c)
e2 = sum(c1')./sum(c')


end

% Функция, осуществляющая копирования графика в буфер обмена 
function CopySaveMenuItem_Callback(x,y,type)
cur_axes = gca;
cmap = get(gcf, 'colormap');
newFig = figure('visible','off');
copyobj(cur_axes,newFig);
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
