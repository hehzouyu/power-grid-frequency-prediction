function feat = vmd_features(f, K)
% VMD_FEATURES  对单条频率曲线做VMD分解,并提取能量熵特征
%
% 大白话:一条频率曲线里"混着"好几种节奏——开头那下快速跌落是一种节奏,
%        后面调频慢慢往回拉是另一种节奏。VMD(变分模态分解)的作用就是
%        把这条曲线拆成 K 个"单一节奏"的分量(专业名叫IMF,本征模态函数),
%        然后我们算每个分量占了多少能量、能量分布有多"散",
%        用几个数字把曲线的形态描述出来——这就叫特征提取。
%
%        能量熵的直观含义:熵越小 = 能量越集中 = 动态过程越单一(比如全是
%        惯量主导的大跌落);熵越大 = 能量越分散 = 各种成分都掺和,
%        动态行为更复杂。
%
% 输入:
%   f : 频率曲线(Hz),1×T 或 T×1 向量(函数内部自动去均值)
%   K : 分解模态数(默认3:对应"快速跌落/中速调节/慢速漂移"三种节奏)
%
% 输出 feat: 结构体
%   imf      : K×T,各模态分量
%   energy   : K×1,各模态能量
%   entropy  : 标量,能量熵(比特)
%   vec      : 1×(K+1) 特征向量 [energy归一化..., entropy]
%
% 依赖: MATLAB Signal Processing Toolbox 的 vmd 函数(R2024b内置)

if nargin < 2, K = 3; end
f = f(:).';                       % 统一成行向量

% --- 关键预处理:去均值 ---
% 大白话:原始频率曲线带一个近50Hz的"大直流"(额定值),而真正的动态信息
%        (跌落、回升)只有零点几个赫兹。直接分解的话,能量几乎全被直流分量
%        吃掉,各IMF能量占比会退化成[1,0,0],特征失去区分度。
%        所以先减掉均值,只对"频率偏差"部分做分解——动态信息全在这里面。
f = f - mean(f);

% --- 第1步:VMD分解 ---
% vmd 返回 K 个模态(MATLAB内置函数,参数名注意是 'NumIMF' 而不是 'NumModes')
% 注意:MATLAB vmd 的输出是【每列一个IMF】(T×K),
%       而本工程约定【每行一个IMF】(K×T),这里做一次方向归一化
imf = vmd(f, 'NumIMF', K, 'AbsoluteTolerance', 1e-6);
if size(imf, 2) == K && size(imf, 1) ~= K
    imf = imf.';     % T×K -> K×T
end

% --- 第2步:各模态能量(即每个IMF平方和,相当于"这个节奏有多强") ---
energy = sum(imf.^2, 2);          % K×1

% --- 第3步:能量熵 ---
% p_i = 第i个模态能量占比;熵 = -Σ p_i·log2(p_i)
p = energy / sum(energy);
p = max(p, 1e-12);                % 防log(0)
entropy = -sum(p .* log2(p));

% --- 第4步:拼装特征向量 ---
% 前半段:各模态能量占比(曲线"能量分布指纹")
% 最后:能量熵(分布"混乱程度")
feat.imf     = imf;
feat.energy  = energy;
feat.entropy = entropy;
feat.vec     = [p.', entropy];
end
