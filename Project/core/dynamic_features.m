function feat = dynamic_features(f, t, t_dist, f_nom)
% DYNAMIC_FEATURES  从频率曲线提取物理动态特征(非泄露版,3维)
%
% 重要修正(特征泄露):旧版输出5维,其中 depth(最大跌落深度)与标签
%   Nadir 存在恒等关系 Nadir = f_nom - depth,等于把答案直接告诉模型,
%   会导致线性回归R²=1.0的假象(实测); rec_rate 由最低点导出,同样泄露。
%   本版只保留与"最低点数值"无恒等关系的3个形态特征。
%
% 各特征的物理含义(大白话):
%   feat(1) RoCoF_max : 扰动后1秒内最大|频率变化率|(Hz/s)
%                     -> 初始RoCoF = dP/(2H),直接反映惯量和扰动强度
%   feat(2) t_nadir   : 从扰动到最低点的时间(s) -> 反映调频响应速度
%                     (注:是"到达时刻",不是"最低点数值",不构成泄露;
%                      但严格早预警场景应只用扰动后前2秒数据,见论文展望)
%   feat(3) f_qss     : 准稳态频率偏差(Hz),末段平均 -> 反映阻尼和调频增益
%
% 输入:
%   f      : 频率曲线(Hz),1×T
%   t      : 时间向量(s),1×T
%   t_dist : 扰动发生时刻(s)
%   f_nom  : 额定频率(Hz,默认50)
%
% 输出:
%   feat : 1×3 特征向量 [RoCoF_max, t_nadir, f_qss]

if nargin < 4, f_nom = 50; end
df = f - f_nom;

idx   = find(t >= t_dist);                 % 扰动后的索引
df_a  = df(idx);  t_a = t(idx);
[~, imin] = min(df_a);                     % 最低点位置(只取时刻,不取数值)

% RoCoF: 扰动后第1秒内的最大变化率
n_win = min(100, numel(df_a)-1);           % 100点 = 1秒
rocof = max(abs(diff(df_a(1:n_win+1))) ./ diff(t_a(1:n_win+1)));

t_nadir = t_a(imin) - t_dist;              % 达谷时间(形态信息)
f_qss   = mean(df(end-50:end));             % 准稳态偏差

feat = [rocof, t_nadir, f_qss];
end
