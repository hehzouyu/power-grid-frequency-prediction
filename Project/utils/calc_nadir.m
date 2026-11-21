function [nadir, t_nadir] = calc_nadir(f, t, sim)
% CALC_NADIR  提取标签:观察截止时刻之后的最低点
%
% 大白话:助教的规矩是"只允许看扰动后前0.5秒,却要预测之后的最低点",
%        所以标签必须在0.5秒截止线之后取,早于此的最低点和任务无关。
%
% 输入:
%   f   : 频率曲线(Hz)
%   t   : 时间向量(s)
%   sim : 仿真参数(需要 t_dist / tau_obs)
%
% 输出:
%   nadir   : 截止后的最低点频率(Hz)
%   t_nadir : 最低点发生的时刻(s)

t_cut = sim.t_dist + sim.tau_obs;      % 观察截止时刻
idx = find(t > t_cut);                  % 严格在截止之后
[nadir, k] = min(f(idx));
t_nadir = t(idx(k));
end
