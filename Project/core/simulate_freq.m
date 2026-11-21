function [t, f, nadir, t_nadir] = simulate_freq(sys, sim)
% SIMULATE_FREQ  执行一次完整的频率动态仿真(TA聚合模型版)
%
% 输入:
%   sys : 系统参数(sys_config)
%   sim : 仿真参数(sim_config)
%
% 输出:
%   t       : 时间向量(s),1×(t_end/dt+1)
%   f       : 频率曲线(Hz),f = f_nom*(1+x)
%   nadir   : 标签——观察截止时刻之后的最低点(Hz)
%             (助教方法学:特征只用前0.5秒,标签取截止后区间,
%              两者时间上不重叠,杜绝信息泄露)
%   t_nadir : 截止后标签最低频率对应的时刻(s)。
%             全局最低点时刻另由 gen_dataset 从完整轨迹计算。

odefun = @(t, y) swing_equation(t, y, sys, sim);
opts = odeset('RelTol', sim.rel_tol);

t_span = 0:sim.dt:sim.t_end;
% 扰动前状态严格平衡。以阶跃时刻的连续状态初始化扰动后积分。
after = t_span > sim.t_dist;
[~, y_after] = ode45(odefun, [sim.t_dist, t_span(after)], [0; 0], opts);
t = t_span;
x = zeros(size(t));
x(after) = y_after(2:end,1).';
f = sys.f_nom * (1 + x);          % 无量纲频偏转回Hz

[nadir, t_nadir] = calc_nadir(f, t, sim);
end
