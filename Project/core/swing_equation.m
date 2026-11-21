function dydt = swing_equation(t, y, sys, sim)
% SWING_EQUATION  二状态聚合模型右端(助教建议形式, 2026-09-24)
%
%   状态 y = [x; p]:
%     x : 无量纲频偏, x = (f - f_nom)/f_nom
%     p : 一次调频实际出力(pu,经调速器惯性环节后的值)
%
%   2H * dx/dt = -d_active - D*x + p
%   Tg * dp/dt = -p + clip(-x/R, 0, p_max)
%
% 符号约定(与助教反馈一致):
%   d_active > 0 表示功率缺额(发电少了),拉低频率 → 取负号
%   p        > 0 表示调频增功,托举频率 → 取正号
%   -D*x     : x<0 时此项为正,负荷减少用电帮回稳 → 天然稳定
%
% 输入:
%   t   : 当前时刻(s)
%   y   : 状态向量 [x; p]
%   sys : 系统参数
%   sim : 仿真参数(需要 t_dist / dP)
%
% 输出:
%   dydt : [dx/dt; dp/dt]

x = y(1);
p = y(2);

% 扰动是否生效:t >= t_dist 后出现功率缺额
if t >= sim.t_dist
    d_active = sim.dP;
else
    d_active = 0;
end

% 调频目标(限幅下垂)
p_target = pfr_model(x, sys);

% 二状态右端
dx = (-d_active - sys.D*x + p) / (2*sys.H);
dp = (p_target - p) / sys.Tg;

dydt = [dx; dp];
end
