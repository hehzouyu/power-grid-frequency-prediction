function [r_sorted, sel_idx] = pearson_select(X, y, r_thresh)
% PEARSON_SELECT  用Pearson相关系数筛选与目标(Nadir)强相关的特征
%
% 大白话:上一步我们提了一堆特征,但不是每个都有用。
%        Pearson相关系数 r 衡量"特征和目标一起涨落"的程度:
%          |r| 接近1 → 强相关,是好特征,留下;
%          |r| 接近0 → 没啥关系,扔掉,免得干扰模型、白添噪声。
%        r 为正 = 特征越大Nadir越大; r 为负 = 特征越大Nadir越小
%        (比如能量占比高→惯量足→跌得浅→Nadir高,就是正相关)。
%
% 输入:
%   X        : N×d 特征矩阵(N条曲线,d个特征)
%   y        : N×1 目标向量(Nadir)
%   r_thresh : |r|筛选阈值(默认0.3,demo阶段;正式可设0.5)
%
% 输出:
%   r_sorted : d×2 矩阵 [特征编号, |r|],按相关性强弱降序
%   sel_idx  : 通过阈值的特征编号

if nargin < 3, r_thresh = 0.3; end

d = size(X, 2);
r = zeros(d, 1);
for j = 1:d
    r(j) = corr(X(:, j), y, 'Type', 'Pearson');   % 逐个特征算相关系数
end

% 排序:按 |r| 从大到小
[~, order] = sort(abs(r), 'descend');
r_sorted = [order, abs(r(order))];

% 筛选
sel_idx = find(abs(r) >= r_thresh);
end
