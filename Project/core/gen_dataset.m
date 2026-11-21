function dataset = gen_dataset(sys, sim, ps)
% GEN_DATASET  批量仿真生成数据集(助教版,2026-09-24)
% 五因素:H, D, R, Tg, dP(无量纲功率缺额)
% 标签:观察截止(t_dist+tau_obs)之后的最低点频率(Hz)
% 事后事件:全局最低点时刻不晚于观察截止的场景,单独标记(助教要求)
%
% 输出 dataset 字段:
%   F       : N×T 频率曲线(Hz)
%   nadir   : N×1 标签——截止后最低点(Hz)
%   t_nadir : N×1 标签最低点发生的时刻(s)
%   params  : N×5 [H, D, R, Tg, dP]
%   posthoc : N×1 逻辑数组,true 表示全局最低点不晚于截止时刻
%   t       : 1×T 时间向量

N = ps.N_case;
T = numel(0:sim.dt:sim.t_end);
t_cut = sim.t_dist + sim.tau_obs;

dataset.F       = zeros(N, T);
dataset.nadir   = zeros(N, 1);
dataset.t_nadir = zeros(N, 1);
dataset.params  = zeros(N, 5);
dataset.posthoc = false(N, 1);
dataset.t       = 0:sim.dt:sim.t_end;

i = 1;
fprintf('开始批量仿真,共 %d 个场景...\n', N);
for iH = 1:numel(ps.H_list)
for iD = 1:numel(ps.D_list)
for iR = 1:numel(ps.R_list)
for iG = 1:numel(ps.Tg_list)
for iP = 1:numel(ps.dP_list)
    s = sys;
    s.H = ps.H_list(iH);  s.D = ps.D_list(iD);
    s.R = ps.R_list(iR);  s.Tg = ps.Tg_list(iG);
    sm = sim;  sm.dP = ps.dP_list(iP);

    [t, f, nad, tnad] = simulate_freq(s, sm);

    % 事后事件判定:全局最低点(扰动后)时刻不晚于观察截止
    idx_g = find(t > sim.t_dist);
    [~, kg] = min(f(idx_g));
    t_global = t(idx_g(kg));

    dataset.F(i, :)      = f;
    dataset.nadir(i)     = nad;
    dataset.t_nadir(i)   = tnad;
    dataset.params(i, :) = [s.H, s.D, s.R, s.Tg, sm.dP];
    dataset.posthoc(i)   = (t_global <= t_cut);
    i = i + 1;
end
end
end
end
end

fprintf('批量仿真完成:%d 条曲线,事后事件 %d 条(%.1f%%)\n', ...
    N, sum(dataset.posthoc), 100*mean(dataset.posthoc));
end
