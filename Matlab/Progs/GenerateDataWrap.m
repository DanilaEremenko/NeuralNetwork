function varargout = GenerateDataWrap(varargin)
%% Задание вариантов исходных распределений классов
if nargin ~= 0
    if strcmp( varargin{1}, 'GenerateData') == 1
        [varargout{1}, varargout{2}] = feval(varargin{1}, varargin{2:end});
    else
        varargout{1} = feval(varargin{1}, varargin{2:end});
    end;
end
end

function class_data = SetDataVariants
global data_type_names axis_rect data_type;
global data_type_per_class C;
% type=1 - нормальное распределение
% par1 - средние значения (строка [c1 c2 ... cn]
% par2 - ковариацоинная матрица

% type=2 - гауссова смесь
% par1 - матрица средних значений компонент (каждая строка - вектор средних [c1 c2 ... cn]
% par2 - 3-мерный массив с ковариацоинными матрицами компонент
% par3 - вектор с коэффициентами pi каждой компоненты

% type=3 - равномерное распределение
% par1 - массив ячеек, каждая из которых характеризует одну из областей,
% где расположено равномерное распределение - {area1, area2, ..., areaN}
% area - описание области в формате {type, par_area1, par_area2, ...}
% par2 - то же, что par1, но описывает области, которные необходимо
% исключить
% Далее приведены примеры описания различных областей
% {1, [x1min x1max x2min x2max], angle} - прямоугольник с точками {x1min, x2min} и {x1max, x2max},
% повернутый относительно центра на угол angle
% {2, [с1 с2], r} - окружность с центром в [c1 c2] радиуса r
% {3, [с1 с2], [r1 r2]} - эллипс с центром в [c1 c2], радиусами r1 и r2,
% повернутый относительно центра на угол angle
% {4, [p11 p12], [p21 p22], [p31 p32], angle} - треугольник с координатами
% вершин {p11, p12}, {p21, p22}, {p31, p32}, повернутый относительно центра
% на угол angle
% {5, xv, yv, angle} - внутренняя часть полигона, заданного точками {xv(i),
% yv(i)}, повернутого относительно центра на угол angle

% type =4 - распределение точек вокруг кривой на плоскости по заданному
% закону (положение на кривой - центр, имеет равномерное распределение
% par1 - описание кривой в форме {curvePar1, curvePar2, ..., curveParN}
% curvePar - описывает часть кривой - возможны следующие варианты
% {1, tstart, tfinish, @fx(t), @fy(t)} - описание в параметрической форме
% x=fx(t), y=fy(t), tstart < t < tfinish
% {2, xv, yv} - описание в форме последовательности координат точек
% {xv{i},yv{i}}
% par2 - описание распределения вокруг кривой в формате {distPar1,
% distPar2, ..., distparN}
% distPar - описывает распределение точек вокруг соответствующей кривой,
% заданной curvePar
% {1, [s1 s2 r]} - нормальное распределение с СКО s1, s2 и к-том корреляции r
% {2, [r1 r2]} - равмномерное распределение в прямоугольнике, отстоящем на
% r1 вправо-влево, r2 - вверх-вниз
% {3, r} - равмномерное распределение в круге радиуса r
% par3 - коэффициенты (вероятности) каждых частей в формате 
% {1} - число точек пропорционально длине кривой
% {2,  [p1 p2 ... pN]} - число точек для каждой кривой задается через pi

data_type = 11; % 1 - нормальное распределение, 2 - смеси, 3 - равномерное
Nmax = 100;
class_data = cell(Nmax,1);
axis_data = cell(Nmax,1);
data_type_names = cell(Nmax,1);

% Классы имеют нормальные распределения

class1.type = 1;
class1.par1 = [1 2];
class1.par2 = eye(2,2);

class2.type = 1;
class2.par1 = [5 5];
class2.par2 = [1 .5; .5 1];

class3.type = 1;
class3.par1 = [4 8];
class3.par2 = eye(2,2);

class_data{1} = { class1, class2, class3 };
axis_data{1} = [-5 10 -2 13];
data_type_names{1} = 'Normal';

% Классы имеют распределения в форме GMM

class1.type = 2;
class1.par1 = [3 2;-2 -3];
class1.par2 = cat(3,[2 -.4;-.4 .5],[1 .5;.5 1]);
class1.par3 = ones(1,2)/2;

class2.type = 2;
class2.par1 = [5 -1;-3 6];
class2.par2 = cat(3,[3 0;0 .5],[3 0;0 3]);
class2.par3 = [0.2 0.7];

class3.type = 2;
class3.par1 = [2 7;-2 1];
class3.par2 = cat(3,[2 -.9;-.9 .5],[2 0;0 1]);
class3.par3 = [0.4 0.6];

class_data{2} = { class1, class2, class3 };
axis_data{2} = [-8 10 -8 13];
data_type_names{2} = 'GMM';

% Классы имеют равномерные распределения

uni_c1_rect = {1, [-6 2 -6 2], 0};
uni_c2_rect = {1, [-4 4 0 10], 0};
uni_c3_circle = {2, [5 2], 4};

class1.type = 3;
class1.par1 = {uni_c1_rect};
class1.par2 = {};

class2.type = 3;
class2.par1 = {uni_c2_rect};
class2.par2 = {};

class3.type = 3;
class3.par1 = {uni_c3_circle};
class3.par2 = {};

class_data{3} = { class1, class2, class3 };
axis_data{3} = [-8 10 -8 13];
data_type_names{3} = 'Uniform';

% Классы имеют различные распределения (смеси, нормальное, равномерное)

class1.type = 1;
class1.par1 = [1 2];
class1.par2 = eye(2,2);

class2.type = 2;
class2.par1 = [5 -1;-3 6];
class2.par2 = cat(3,[3 0;0 .5],[3 0;0 3]);
class2.par3 = [0.2 0.7];

class3.type = 3;
class3.par1 = {{4, [-7 0], [-6 4], [-4 2], 15},  {2, [5 5], 3}, {3, [5 -5], [4 1], -20} };
class3.par2 = {};

class4.type = 3;
class4.par1 = {{2, [4 10], 2}, {1, [-7 0 -6 0], 20}, {5, [6 7 8 6 8 6], [-4 -1 -4 -2 -2 -4], -10}};
class4.par2 = {{1,[-5 -2 -5 -1], 20}};

class_data{4} = { class1, class2, class3, class4 };
axis_data{4} = [-8 10 -8 13];
data_type_names{4} = 'Norm+GMM+Uniform';

% Классы имеют равномерные распределения в форме концентрических
% окружностей
class1.type = 3;
class1.par1 = {{2, [2 2], 2}};
class1.par2 = {};

class2.type = 3;
class2.par1 = {{2, [2 2], 4}};
class2.par2 = {{2, [2 2], 2}};

class3.type = 3;
class3.par1 = {{2, [2 2], 6}};
class3.par2 = {{2, [2 2], 4}};

class_data{5} = { class3, class2, class1 };
axis_data{5} = [-8 10 -8 13];
data_type_names{5} = 'Conc. Circles';

% Классы имеют равномерные распределения в форме горизонтальных полос
class1.type = 3;
class1.par1 = {{1, [1 7 1 3], 0}};
class1.par2 = {};

class2.type = 3;
class2.par1 = {{1, [1 7 3 5], 0}};
class2.par2 = {};

class3.type = 3;
class3.par1 = {{1, [1 7 5 7], 0}};
class3.par2 = {};

class_data{6} = { class1, class2, class3 };
axis_data{6} = [0 8 0 8];
data_type_names{6} = 'Hor. Stripes';


% Классы имеют равномерные распределения в форме вертикальных полос
class1.type = 3;
class1.par1 = {{1, [1 3 1 7], 0}};
class1.par2 = {};

class2.type = 3;
class2.par1 = {{1, [3 5 1 7], 0}};
class2.par2 = {};

class3.type = 3;
class3.par1 = {{1, [ 5 7 1 7], 0}};
class3.par2 = {};

class_data{7} = { class1, class2, class3 };
axis_data{7} = [0 8 0 8];
data_type_names{7} = 'Ver. Stripes';

% Классы имеют равномерные распределения в форме 4 квадратов
class1.type = 3;
class1.par1 = {{1, [1 3 1 3], 0}};
class1.par2 = {};

class2.type = 3;
class2.par1 = {{1, [3 5 1 3], 0}};
class2.par2 = {};

class3.type = 3;
class3.par1 = {{1, [ 1 3 3 5], 0}};
class3.par2 = {};

class4.type = 3;
class4.par1 = {{1, [ 3 5 3 5], 0}};
class4.par2 = {};

class_data{8} = { class1, class2, class3, class4 };
axis_data{8} = [0 6 0 6];
data_type_names{8} = 'Squares';

% Классы имеют равномерные распределения в форме 4 квадратов
circle_data = {{2, [0 0], 1}};
p = { [-2 -2],[0 -2],[2 -2], [2 0],[2 2],[0 2],[-2 2],[-2 0],[0 0] }; 

class.type = 3;
class.par1 = circle_data;
class.par2 = {};

class_data{9} = { class };
for i = 1:length(p) - 1
    class.type = 3;
    if i == length(p)-1
        class.par1 = {{4, p{i}, p{1}, p{length(p)},0}};
    else
        class.par1 = {{4, p{i}, p{i+1}, p{length(p)},0}};
    end;        
    class.par2 = circle_data;
    class_data{9} = [class_data{9}, class ];
end;
axis_data{9} = [-3 3 -3 3];
data_type_names{9} = 'Circle+Lines';

% Задание в форме точек, распределенных вокруг кривой
class1.type = 4; 
fx = @(t, k1, k2)t.*cos(k1*t+k2); fy = @(t, k1, k2)t.*sin(k1*t+k2);
tmin = 0; tmax = 5; dt = .2;
k1 = 2; k2 = 0; 
s1 = .14; s2 = .14; r = 0;
class1.par1 = {{2, fx(tmin:dt:tmax, k1, k2), fy(tmin:dt:tmax, k1, k2)}};  % {2, xv, yv}
class1.par2 = {{1, [s1 s2 r]}};     % {1, [s1 s2 r]} - нормальное распределение с СКО s1, s2 и к-том корреляции r
class1.par3 = {1};

class2.type = 4; 
k2 = 2;
r = 0.2;
class2.par1 = {{1, tmin, tmax, @(x)fx(x,k1,k2), @(x)fy(x,k1,k2)}}; % {1, tstart, tfinish, @fx(t), @fy(t)}
class2.par2 =  {{3, r}}; % равмномерное распределение в круге радиуса r
class2.par3 = {1};

class3.type = 4; 
k2 = 4;
r1 = 0.25; r2 = 0.25;
class3.par1 = {{1, tmin, tmax, @(x)fx(x,k1,k2), @(x)fy(x,k1,k2)}}; % {1, tstart, tfinish, @fx(t), @fy(t)}
class3.par2 = {{2, [r1 r2]}};     % {2, [r1 r2]} - равмномерное распределение в прямоугольнике, отстоящем на
class3.par3 = {1};

class_data{10} = {class1, class2, class3};
axis_data{10} = [-5 5 -5 5];
data_type_names{10} = 'Spiral';

% Один из вариантов мозайки

h = 1;
p_start = [0; 0];
n1 = 6;
n2 = 6;
hex_pgm = generate_hex_pgm(h, p_start, n1, n2);
[ncl, t_pgm] = tile_hex_pgm (hex_pgm, 1);
class_data{11} = {};

for c = 1:ncl
    class.type = 3;
    class.par1 = {};
    i_c = find(t_pgm == c);
    for i = 1:length(i_c)
        ii = i_c(i);
        class.par1{i} = {5, hex_pgm{ii}(1,:), hex_pgm{ii}(2,:), 0}; %{5, xv, yv, angle}
    end;
    class.par2 = {};
    class_data{11} = [class_data{11}, class ];
end;
axis_data{11} = [0 6 0 6];
data_type_names{11} = 'Hex.Mozaic';

plot_temp = 0;
if plot_temp
    colors = {'r','g','b','y'};
    for i = 1:n1
        for j = 1:n2
            plot(hex_pgm{i,j}(1,:), hex_pgm{i,j}(2,:));
            patch(hex_pgm{i,j}(1,:),hex_pgm{i,j}(2,:),colors{t_pgm(i,j)});
            hold on;
        end;
    end;
    axis([0 h*sqrt(3)*(n1-1) 0 3*h*(n2-1)/2]);
end;

class_data = class_data(~cellfun('isempty',class_data));
axis_data = axis_data(~cellfun('isempty',axis_data));
data_type_names = data_type_names(~cellfun('isempty',data_type_names));

axis_rect = axis_data;

data_type_per_class = cell(length(class_data),1);
C = zeros(1, length(class_data));
for i = 1:length(class_data)
    C(i) = length(class_data{i});
    for j = 1:length(class_data{i})
        data_type_per_class{i}(j) = class_data{i}{j}.type;
    end;
end;


% Данные, описывающие рапределение исходных классов
%class_data = {class_data1, class_data2, class_data3, class_data4, class_data5, class_data6, class_data7, class_data8, class_data9, class_data10};

end

function hex_poly_matr = generate_hex_pgm(h, p_start, n1, n2)
k = h*sqrt(3)/2;
hex_poly = [0 k k 0 -k -k 0;-h -h/2 h/2 h h/2 -h/2 -h];
hex_poly = hex_poly + p_start * ones(1, size(hex_poly,2));

hex_poly_matr = cell(n1, n2);

for i = 1:n1
    for j = 1:n2
        hex_poly_matr{i,j} = hex_poly + [(i-1)*2*k + mod(j-1,2) * k;(j-1)*3/2*h] * ones(1, size(hex_poly,2));
    end;
end;

end
function [ncl, t_pgm] = tile_hex_pgm (pgm, type)
[n1,n2] = size(pgm);
t_pgm = zeros(n1,n2);
ncl = 4;

f_ind = @(i,j)(mod(j+1,2)==0&&mod(j+1,4)~=0 && mod(i+1,2)==0) || (mod(j+1,4) == 0 && mod(i,2)==0);

for i = 1:n1
    for j = 1:n2
        t_pgm(i,j) =    [1:4]*[f_ind(i,j);f_ind(i+1,j); f_ind(i,j+1); f_ind(i+1,j+1)];
    end;
end;

end

% Генерация данных для основной задачи классификации (нормальное
% распределение)
function [dataAll, groupsAll] = GenerateData(data_type, N_data, n_class_2)
global class_data;

class_data_cur = class_data{data_type};

for c = 1:length(n_class_2)
    c1 = n_class_2(c);
    class_cur = class_data_cur{c1};
    
    if class_cur.type == 1
        data{c} = mvnrnd(class_cur.par1, class_cur.par2, N_data(c));
    elseif class_cur.type == 2
        obj = gmdistribution(class_cur.par1, class_cur.par2, class_cur.par3);
        data{c} = random(obj,N_data(c));
    elseif class_cur.type == 3
        data{c} = uni_geom_rnd(class_cur.par1,class_cur.par2,N_data(c));
    elseif class_cur.type == 4
        data{c} = curve_rnd(class_cur.par1,class_cur.par2, class_cur.par3, N_data(c));
    end;
end;

dataAll = [];groupsAll = [];

for c = 1:length(n_class_2)
    dataAll = [dataAll; data{c}];
    groupsAll = [groupsAll c*ones(1, N_data(c))];
end;
end

%% Генерация равномерных распределений внутри геометрических фигур
function y = uni_geom_rnd(uni_data_add, uni_data_remove, N)
N_int = 1e4;
[rect, f_val, sq] = find_uni_pdf(uni_data_add, uni_data_remove, N_int);
if N < 200
    k_Ng = 5;
else
    k_Ng = 3;
end;
Ng = ceil(N*f_val*sq*k_Ng);

x1min = rect(1); x1max = rect(2); x2min = rect(3); x2max = rect(4);
y = ones(Ng,1)*[x1min x2min] + rand(Ng,2)*[x1max-x1min 0; 0 x2max-x2min];

x = zeros(Ng,1);
for i = 1:length(uni_data_add)
    x = x | check_geom(y, uni_data_add{i});
end;
for i = 1:length(uni_data_remove)
    x = x & ~check_geom(y, uni_data_remove{i});
end;

y = y(x~=0,:);
y = y(1:N,:);
end
% Плотность равмноерного распределения внутри геометрической области
function y = uni_geom_pdf(x, uni_data_add, uni_data_remove)
% Находим вначале прямоугольник с областью, включающей в себя все
% под-области
N_int = 1e4;
[rect, f_val, sq] = find_uni_pdf(uni_data_add, uni_data_remove, N_int);
y = zeros(length(x),1);

for i = 1:length(uni_data_add)
    y = y | check_geom(x, uni_data_add{i});
end;
for i = 1:length(uni_data_remove)
    y = y & ~check_geom(x, uni_data_remove{i});
end;
y = y * f_val;
end
% Плотность распределения вокруг кривой (принимается равномерной)
function y = curve_rnd(curve_param, distr_param, p_param, N)
N_curves = length(distr_param);
% Определение числа точек для каждой кривой
if p_param{1} == 1
    % {1} - число точек пропорционально длине кривой
    % Вычисление длин кривых
    Np_calc_length = 200;
    for i = 1:N_curves
        [xx,yy] = curve_get_points(curve_param{i}, Np_calc_length);
        len_ci(i) = sum(abs(diff(complex(xx,yy))));
    end;
    Ni = len_ci/sum(len_ci)*N;
elseif p_param == 2
    % {2,  [p1 p2 ... pN]} - число точек для каждой кривой задается через pi
    Ni = p_param{2}*N;
    Ni(end) = N - sum(Ni(1:end-1));
end;

yy = cell(length(curve_param),1);
for i = 1:N_curves
   [x,y] =  curve_get_points(curve_param{i}, Ni(i));
   yy{i} = [x' y'];
end;

for i = 1:N_curves
    distr_part_par = distr_param{i};
    if distr_part_par{1} == 1 
        s1 = distr_part_par{2}(1); s2 = distr_part_par{2}(2); r = distr_part_par{2}(3);
        % {1, [s1 s2 r]} - нормальное распределение с СКО s1, s2 и к-том корреляции r
        dy = mvnrnd([0 0], [s1^2 r*s1*s2; r*s1*s2 s2^2], Ni(i));
    elseif distr_part_par{1} == 2
        % {2, [r1 r2]} - равмномерное распределение в прямоугольнике, отстоящем на r1 вправо-влево, r2 - вверх-вниз
        r1 = distr_part_par{2}(1); r2 = distr_part_par{2}(2);
        dy = ones(Ni(i),1)*[-r1 -r2] + rand(Ni(i),2)*[2*r1 0; 0 2*r2];
    elseif distr_part_par{1} == 3
        % {3, r} - равмномерное распределение в круге радиуса r
        r = distr_part_par{2}(1);
        r_i = rand(Ni(i),1)*r;
        phi_i = rand(Ni(i),1)*2*pi;
        dy = [r_i.*cos(phi_i) r_i.*sin(phi_i)];
        
    end;
    yy{i} = yy{i} + dy;
end;

y = zeros(N,2); ct = 1;
for i = 1:N_curves
    y(ct:ct+Ni(i)-1,:) = yy{i}; ct = ct + Ni(i);
end;
end
function [x,y] = curve_get_points(curve_part_par, N)

if curve_part_par{1} == 1
    % {1, tstart, tfinish, @fx(t), @fy(t)} - описание в параметрической форме x=fx(t), y=fy(t), tstart < t < tfinish
    t_start = curve_part_par{2};
    t_finish = curve_part_par{3};
    tt = linspace(t_start, t_finish, N);
    fx = curve_part_par{4};
    fy = curve_part_par{5};
    x = fx(tt);
    y = fy(tt);
elseif curve_part_par{1} == 2
    xx = curve_part_par{2};
    yy = curve_part_par{3};
    
    if length(xx) == 1
        x = xx*ones(N,1);
        y = yy*ones(N,1);
    else
        % Интерполяция сплайнами
        tt = linspace(0,1, length(xx));
        pp = spline(tt,[xx;yy]);
        tt2 = linspace(0,1, N);
        zz = ppval(pp, tt2);
        x = zz(1,:);
        y = zz(2,:);
%        plot(x,y); hold on; plot(xx,yy, 'or');
        
        % {2, xv, yv} - описание в форме последовательности координат точек {xv{i},yv{i}}
    end;
end;
end

function y = curve_pdf(x, curve_param, distr_param, p_param)
% Находим вначале прямоугольник с областью, включающей в себя все
% под-области
N_int = 1e4;
N_curves = length(distr_param);
[~, f_val, ~] = find_curve_pdf(curve_param, distr_param, p_param, N_int);

y = zeros(length(x),1);

for i = 1:N_curves
    y = y | check_curve(x, curve_param{i}, distr_param{i}, p_param{i});
end;

y = y * f_val;
end
% Вспомогательная функция для приближенного определения плотности
% равномерного распределения по площади
function [rect, f_val, sq] = find_uni_pdf(uni_data_add, uni_data_remove, N_int)
x1min = inf; x2min = inf; x1max = -inf; x2max = -inf;
for i = 1:length(uni_data_add)
    uni_data_i = uni_data_add{i};
    [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect(uni_data_i);
    x1min = min([x1min x1min_g]);
    x1max = max([x1max x1max_g]);
    x2min = min([x2min x2min_g]);
    x2max = max([x2max x2max_g]);
end;
x_int = ones(N_int,1)*[x1min x2min] + rand(N_int,2)*[x1max-x1min 0; 0 x2max-x2min];
y_int = zeros(length(x_int),1);

for i = 1:length(uni_data_add)
    y_int = y_int | check_geom(x_int, uni_data_add{i});
end;
for i = 1:length(uni_data_remove)
    y_int = y_int & ~check_geom(x_int, uni_data_remove{i});
end;

rect = [x1min x1max x2min x2max];
sq = (x1max-x1min)*(x2max-x2min);
sq1 = sq*sum(y_int)/length(x_int);
f_val = 1/sq1;
end
function [rect, f_val, sq] = find_curve_pdf(curve_param, distr_param, p_param, N_int)
global axis_rect data_type;
N_curves = length(distr_param);

rect = axis_rect{data_type}
x1min = rect(1);
x1max = rect(2);
x2min = rect(3);
x2max = rect(4);

x_int = ones(N_int,1)*[x1min x2min] + rand(N_int,2)*[x1max-x1min 0; 0 x2max-x2min];
y_int = zeros(length(x_int),1);

for i = 1:N_curves
    y_int = y_int | check_curve(x_int, curve_param{i}, distr_param{i}, p_param{i});
end;

sq = (x1max-x1min)*(x2max-x2min);
sq1 = sq*sum(y_int)/length(x_int);
f_val = 1/sq1;
end
% Функции определения попадания в геометрическую фигуру, заданную
% координатами и параметрами
function y = check_geom(x_int, uni_data_i)
if uni_data_i{1} == 1
    y = check_rect(x_int, uni_data_i{2}, uni_data_i{3});
elseif uni_data_i{1} == 2
    y = check_circle(x_int, uni_data_i{2}, uni_data_i{3});
elseif uni_data_i{1} == 3
    y = check_ellipse(x_int, uni_data_i{2}, uni_data_i{3}, uni_data_i{4});
elseif uni_data_i{1} == 4
    y = check_triangle(x_int, uni_data_i{2}, uni_data_i{3}, uni_data_i{4}, uni_data_i{5});
elseif uni_data_i{1} == 5
    y = check_polygon(x_int, uni_data_i{2}, uni_data_i{3}, uni_data_i{4});
end;
end
function y = check_rect(x, rect_data, alpha)
alpha_rad = pi/180*alpha;
center = [(rect_data(1)+rect_data(2))/2 (rect_data(3)+rect_data(4))/2];
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
x2 = (x - ones(size(x,1),1)*center) * Mrot + ones(size(x,1),1)*center;
y = x2(:,1) >= rect_data(1) & x2(:,1) <= rect_data(2) & x2(:,2) >= rect_data(3) & x2(:,2) <= rect_data(4);
end
function y = check_circle(x, center, r)
y = (x(:,1)-center(1)).^2 + (x(:,2)-center(2)).^2 <= r^2;
end
function y = check_ellipse(x, center, r, alpha)
alpha_rad = pi/180*alpha;
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
x = (x - ones(size(x,1),1)*center) * Mrot + ones(size(x,1),1)*center;
y = (x(:,1)-center(1)).^2/r(1)^2 + (x(:,2)-center(2)).^2/r(2)^2 <= 1;
end
function y = check_triangle(x, p1, p2, p3, alpha)
% если сумма расстояний от каждой вершины до точки меньше периметра, то внутри, иначе - вовне
%per = sum((p1 - p2).^2,2).^0.5 + sum((p1 - p3).^2,2).^0.5 + sum((p2 - p3).^2,2).^0.5;
%ind = sum((x - ones(length(x),1)*p1).^2, 2).^0.5 + sum((x - ones(length(x),1)*p2).^2,2).^0.5 + sum((x - ones(length(x),1)*p3).^2,2).^0.5;
%y = ind < per
alpha_rad = pi/180*alpha;
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
center = 1/3*(p1+p2+p3);
x = (x - ones(size(x,1),1)*center) * Mrot + ones(size(x,1),1)*center;
y = inpolygon(x(:,1), x(:,2), [p1(1) p2(1) p3(1) p1(1)], [p1(2) p2(2) p3(2) p1(2)]);
end
function y = check_polygon(x, xv, yv, alpha)
% если сумма расстояний от каждой вершины до точки меньше периметра, то внутри, иначе - вовне
%per = sum((p1 - p2).^2,2).^0.5 + sum((p1 - p3).^2,2).^0.5 + sum((p2 - p3).^2,2).^0.5;
%ind = sum((x - ones(length(x),1)*p1).^2, 2).^0.5 + sum((x - ones(length(x),1)*p2).^2,2).^0.5 + sum((x - ones(length(x),1)*p3).^2,2).^0.5;
%y = ind < per
alpha_rad = pi/180*alpha;
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
center = 1/3*[sum(xv(1:end-1)) sum(yv(1:end-1))];
x = (x - ones(size(x,1),1)*center) * Mrot + ones(size(x,1),1)*center;
y = inpolygon(x(:,1), x(:,2), xv, yv);
end
function y = check_curve(x, curve_param, distr_param, p_param)
Np_calc_length = 200;
%minD = cell(N_curves,1);
y = zeros(size(x,1),1);
[xx,yy] = curve_get_points(curve_param, Np_calc_length);

if distr_param{1} == 1
    D = pdist2(x, [xx' yy']);
    minD = min(D,[],2);
    s1 = distr_param{2}(1); s2 = distr_param{2}(2); r = distr_param{2}(3);
    y = y | minD < 3*sqrt(s1^2+s2^2);
elseif distr_param{1} == 2
    eps = 0;
    r1 = distr_param{2}(1); r2 = distr_param{2}(2);
    Dx = pdist2(x(:,1), xx');
    fx = Dx+eps <= r1;
    Dy = pdist2(x(:,2), yy');
    fy = Dy+eps <= r2;
    fxy = fx & fy;
    y = y | sum(fxy')' > 0;
elseif distr_param{1} == 3
    D = pdist2(x, [xx' yy']);
    minD = min(D,[],2);
    r = distr_param{2}(1);
    y = y | minD < r;
end;

end
% Функции определения прямоугольника, включающего в себя заданную
% геометрическую фигуру
function [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect(uni_data_i)
    if uni_data_i{1} == 1
        [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_rect(uni_data_i{2}, uni_data_i{3});
    elseif uni_data_i{1} == 2
        [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_circle(uni_data_i{2}, uni_data_i{3});
    elseif uni_data_i{1} == 3
        [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_ellipse(uni_data_i{2}, uni_data_i{3}, uni_data_i{4});
    elseif uni_data_i{1} == 4
        [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_form_triangle(uni_data_i{2}, uni_data_i{3}, uni_data_i{4}, uni_data_i{5});
    elseif uni_data_i{1} == 5
        [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_form_polygon(uni_data_i{2}, uni_data_i{3}, uni_data_i{4});
    end;
end
function [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_rect(rect, alpha)
alpha_rad = -pi/180*alpha;
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
center = [(rect(1)+rect(2))/2 (rect(3)+rect(4))/2];

p_corn = [rect(1) rect(3); rect(1) rect(4); rect(2) rect(3); rect(2) rect(4)];
p_corn = (p_corn - ones(size(p_corn,1),1)*center) * Mrot + ones(size(p_corn,1),1)*center;
x1min_g = min(p_corn(:,1)); x1max_g = max(p_corn(:,1)); x2min_g = min(p_corn(:,2)); x2max_g = max(p_corn(:,2));
end
function [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_circle(center, r)
x1min_g = center(1) - r; x1max_g = center(1) + r; x2min_g = center(2) - r; x2max_g = center(2) + r;
end
function [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_ellipse(center, r, alpha)
x1min_g = center(1) - r(1); x1max_g = center(1) + r(1); x2min_g = center(2) - r(2); x2max_g = center(2) + r(2);
[x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_rect([x1min_g,x1max_g,x2min_g, x2max_g], alpha);
end
function [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_form_triangle(p1, p2, p3, alpha)
alpha_rad = -pi/180*alpha;
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
center = 1/3*(p1+p2+p3);

p_corn = [p1; p2; p3];
p_corn = (p_corn - ones(size(p_corn,1),1)*center) * Mrot + ones(size(p_corn,1),1)*center;
x1min_g = min(p_corn(:,1)); x1max_g = max(p_corn(:,1)); x2min_g = min(p_corn(:,2)); x2max_g = max(p_corn(:,2));
end
function [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_form_polygon(xv, yv, alpha)
alpha_rad = -pi/180*alpha;
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
center = 1/3*[sum(xv(1:end-1)) sum(yv(1:end-1))];

p_corn = [xv' yv'];
p_corn = (p_corn - ones(size(p_corn,1),1)*center) * Mrot + ones(size(p_corn,1),1)*center;
x1min_g = min(p_corn(:,1)); x1max_g = max(p_corn(:,1)); x2min_g = min(p_corn(:,2)); x2max_g = max(p_corn(:,2));
end
