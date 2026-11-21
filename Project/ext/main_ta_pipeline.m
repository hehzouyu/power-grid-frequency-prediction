function main_ta_pipeline()
% MAIN_TA_PIPELINE  助教版完整主线实验(2026-09-24, 一键运行)
%
% 流程:数据集→事后事件单列→60/20/20划分→
%       基线(持续性)→岭回归(λ验证集)→提升树→
%       VMD消融(同划分同回归器)→窗口长度(0.25/0.5/1.0s)→
%       噪声(只加输入,目标无噪)→外推检验(H=6)→分组报告(按H/d)
%
% 输出: results/tables/ta_*.csv
% 预计耗时: 约20~40分钟(数据生成10分钟+VMD特征约15分钟)

t_start = tic;
fprintf('===== 助教版主线实验开始 =====\n');
sys = sys_config();
sim = sim_config();

%% ===== 1. 数据集(已存在则复用) =====
mat_ta = 'data/raw/dataset_ta.mat';
if ~exist(mat_ta, 'file')
    if ~exist('data/raw', 'dir'), mkdir('data/raw'); end
    ps = param_space('full');
    dataset = gen_dataset(sys, sim, ps);
    save(mat_ta, 'dataset', 'sys', 'sim', 'ps');
else
    load(mat_ta);
end
N = size(dataset.F, 1);
t_cut = sim.t_dist + sim.tau_obs;          % 观察截止
main_mask = ~dataset.posthoc;              % 主评估:非事后事件
fprintf('数据集: %d 条 | 事后事件: %d 条(%.1f%%,单列处理)\n', ...
    N, sum(dataset.posthoc), 100*mean(dataset.posthoc));

%% ===== 2. 60/20/20 划分(按场景,种子42) =====
rng(42);
idx_all = find(main_mask);
idx = idx_all(randperm(numel(idx_all)));
n_tr = floor(0.6*numel(idx));  n_va = floor(0.2*numel(idx));
idx_tr = idx(1:n_tr);
idx_va = idx(n_tr+1 : n_tr+n_va);
idx_te = idx(n_tr+n_va+1 : end);
fprintf('划分(非事后事件): 训练 %d / 验证 %d / 测试 %d\n', ...
    numel(idx_tr), numel(idx_va), numel(idx_te));

%% ===== 3. 主窗口(W=0.5s)特征提取 =====
% 助教规定的手工特征:截止时刻频偏、窗口平均斜率、波动幅度
W0 = 0.5;
fprintf('\n--- 提取主窗口特征 W=%.2fs ---\n', W0);
% VMD模态数K由验证集从{2,3,4}中选出(K=4验证集RMSE最小,见论文3.2节)
[PhiB, PhiV] = ta_features(dataset, sim, sys, W0, 4);
% 标签:随窗口自适应(截止后最低点),主窗口下与数据集标签一致
y_lab = ta_label(dataset, sim, W0);

% 标准化(各自用训练集统计量)
Zb = (PhiB - mean(PhiB(idx_tr,:),1)) ./ std(PhiB(idx_tr,:),1);
Zv = (PhiV - mean(PhiV(idx_tr,:),1)) ./ std(PhiV(idx_tr,:),1);
% VMD分解失败(NaN)的行,用训练集均值填充,避免污染回归
bad = any(isnan(Zv), 2);
if any(bad)
    Zv(bad, :) = repmat(mean(Zv(idx_tr,:),1), sum(bad), 1);
end

%% ===== 4. 四个基础模型评估 =====
fprintf('\n--- 基础模型对比(测试集,非事后事件) ---\n');
rows = {};
% (a) 持续性基线:用截止时刻频率保持值预测
y_base = dataset.F(idx_te, find(dataset.t >= t_cut, 1));
m = calc_metrics(y_lab(idx_te), y_base);  m.time_s = 0;
rows = add_row(rows, '持续性基线(截止时刻频率)', '-', m);
fprintf('持续性基线: RMSE=%.5f MAE=%.5f P95=%.5f R2=%.4f\n', m.rmse, m.mae, m.p95, m.r2);
% (b) 解析法基线
tic; y_ana = baseline_analytic(dataset.params(idx_te,:), sys); m = calc_metrics(y_lab(idx_te), y_ana);
m.time_s = toc;
rows = add_row(rows, '解析法基线(准稳态近似)', '-', m);
fprintf('解析法基线: RMSE=%.5f MAE=%.5f P95=%.5f R2=%.4f\n', m.rmse, m.mae, m.p95, m.r2);
% (c) 岭回归(手工三特征)
tic;
[mdl_ridge, lam_best] = ridge_train(Zb(idx_tr,:), y_lab(idx_tr), Zb(idx_va,:), y_lab(idx_va));
y_ridge = [ones(numel(idx_te),1), Zb(idx_te,:)] * mdl_ridge.theta;
m_ridge = calc_metrics(y_lab(idx_te), y_ridge);  m_ridge.time_s = toc;
rows = add_row(rows, '岭回归(手工特征)', sprintf('lambda=%.4g', lam_best), m_ridge);
fprintf('岭回归:     RMSE=%.5f MAE=%.5f P95=%.5f R2=%.4f (λ=%.4g)\n', ...
    m_ridge.rmse, m_ridge.mae, m_ridge.p95, m_ridge.r2, lam_best);
% (d) 提升树(手工三特征, 同特征同划分)
tic;
mdl_bst = fitrensemble(Zb(idx_tr,:), y_lab(idx_tr), 'Method', 'LSBoost', ...
    'NumLearningCycles', 100, 'LearnRate', 0.1);
y_bst = predict(mdl_bst, Zb(idx_te,:));
m_bst = calc_metrics(y_lab(idx_te), y_bst);  m_bst.time_s = toc;
rows = add_row(rows, '提升树(手工特征)', 'LSBoost,100棵,lr=0.1', m_bst);
fprintf('提升树:     RMSE=%.5f MAE=%.5f P95=%.5f R2=%.4f\n', ...
    m_bst.rmse, m_bst.mae, m_bst.p95, m_bst.r2);

T_main = rows2table(rows);
writetable(T_main, 'results/tables/ta_main_results.csv');

%% ===== 5. VMD消融(同划分,同回归器,只加VMD特征) =====
fprintf('\n--- VMD消融(特征=[手工+VMD],与手工特征对比) ---\n');
Z_bv = [Zb, Zv];
tic;
[mdl_r2, lam2] = ridge_train(Z_bv(idx_tr,:), y_lab(idx_tr), Z_bv(idx_va,:), y_lab(idx_va));
y_r2 = [ones(numel(idx_te),1), Z_bv(idx_te,:)] * mdl_r2.theta;
m_r2 = calc_metrics(y_lab(idx_te), y_r2);  m_r2.time_s = toc;
tic;
mdl_b2 = fitrensemble(Z_bv(idx_tr,:), y_lab(idx_tr), 'Method', 'LSBoost', ...
    'NumLearningCycles', 100, 'LearnRate', 0.1);
y_b2 = predict(mdl_b2, Z_bv(idx_te,:));
m_b2 = calc_metrics(y_lab(idx_te), y_b2);  m_b2.time_s = toc;

rows = {};
rows = add_row(rows, '岭回归(仅手工特征)', sprintf('lambda=%.4g', lam_best), m_ridge);
rows = add_row(rows, '岭回归(手工+VMD)', sprintf('lambda=%.4g', lam2), m_r2);
rows = add_row(rows, '提升树(仅手工特征)', 'LSBoost,100棵,lr=0.1', m_bst);
rows = add_row(rows, '提升树(手工+VMD)', 'LSBoost,100棵,lr=0.1', m_b2);
T_abl = rows2table(rows);
writetable(T_abl, 'results/tables/ta_ablation.csv');
fprintf('岭回归+VMD:   R2=%.4f | 提升树+VMD:   R2=%.4f\n', m_r2.r2, m_b2.r2);

%% ===== 6. 窗口长度对比(0.25/0.5/1.0s,提升树) =====
fprintf('\n--- 窗口长度对比(提升树) ---\n');
rows = {};
for W = [0.25, 0.5, 1.0]
    [PhiB_w, ~] = ta_features(dataset, sim, sys, W, 3);
    y_w = ta_label(dataset, sim, W);
    Zw = (PhiB_w - mean(PhiB_w(idx_tr,:),1)) ./ std(PhiB_w(idx_tr,:),1);
    tic;
    mdl_w = fitrensemble(Zw(idx_tr,:), y_w(idx_tr), 'Method', 'LSBoost', ...
        'NumLearningCycles', 100, 'LearnRate', 0.1);
    yhat = predict(mdl_w, Zw(idx_te,:));
    m_w = calc_metrics(y_w(idx_te), yhat);  m_w.time_s = toc;
    rows = add_row(rows, sprintf('W=%.2fs', W), '提升树', m_w);
    fprintf('W=%.2fs: RMSE=%.5f R2=%.4f\n', W, m_w.rmse, m_w.r2);
end
T_win = rows2table(rows);
writetable(T_win, 'results/tables/ta_window.csv');

%% ===== 7. 噪声鲁棒性(只加输入,目标无噪,提升树,W=0.5) =====
fprintf('\n--- 噪声鲁棒性(输入加噪,目标无噪) ---\n');
rows = {};
for snr = [40, 30, 20]
    rng(42);
    F_noisy = dataset.F;
    for i = 1:N
        F_noisy(i,:) = add_noise(dataset.F(i,:), snr, sys.f_nom);
    end
    ds_n = dataset;  ds_n.F = F_noisy;
    [PhiB_n, ~] = ta_features(ds_n, sim, sys, W0, 3);
    Zn = (PhiB_n - mean(PhiB_n(idx_tr,:),1)) ./ std(PhiB_n(idx_tr,:),1);
    mdl_n = fitrensemble(Zn(idx_tr,:), y_lab(idx_tr), 'Method', 'LSBoost', ...
        'NumLearningCycles', 100, 'LearnRate', 0.1);
    yhat_n = predict(mdl_n, Zn(idx_te,:));
    m_n = calc_metrics(y_lab(idx_te), yhat_n);  m_n.time_s = NaN;
    rows = add_row(rows, sprintf('SNR=%ddB', snr), '提升树', m_n);
    fprintf('SNR=%ddB: RMSE=%.5f R2=%.4f\n', snr, m_n.rmse, m_n.r2);
end
T_noise = rows2table(rows);
writetable(T_noise, 'results/tables/ta_noise.csv');

%% ===== 8. 外推检验(训练H<=5,测试H=6) =====
fprintf('\n--- 外推检验(训练H<=5,测试H=6) ---\n');
ext_mask = main_mask & (dataset.params(:,1) == 6);
tr_mask  = main_mask & (dataset.params(:,1) <= 5);
idx_e = find(ext_mask);  idx_tre = find(tr_mask);
mdl_e = fitrensemble(Zb(idx_tre,:), y_lab(idx_tre), 'Method', 'LSBoost', ...
    'NumLearningCycles', 100, 'LearnRate', 0.1);
y_e = predict(mdl_e, Zb(idx_e,:));
m_e = calc_metrics(y_lab(idx_e), y_e);  m_e.time_s = NaN;
T_ext = table({'H=6(训练未见)'}, m_e.rmse, m_e.mae, m_e.p95, m_e.r2, ...
    'VariableNames', {'Extrapolation', 'RMSE_Hz', 'MAE_Hz', 'P95_Hz', 'R2'});
writetable(T_ext, 'results/tables/ta_extrapolation.csv');
fprintf('外推H=6: RMSE=%.5f R2=%.4f\n', m_e.rmse, m_e.r2);

%% ===== 9. 分组报告(按H、按d,提升树,W=0.5) =====
fprintf('\n--- 分组报告(测试集,按H和d分组) ---\n');
yhat_all = nan(N,1);  yhat_all(idx_te) = y_bst;
rows = {};
for g = 1:numel(ps.H_list)
    mm = idx_te(dataset.params(idx_te,1) == ps.H_list(g));
    if numel(mm) < 3, continue; end
    m_g = calc_metrics(y_lab(mm), yhat_all(mm));
    rows = add_row(rows, sprintf('H=%gs', ps.H_list(g)), '按H分组', m_g);
end
for g = 1:numel(ps.dP_list)
    mm = idx_te(abs(dataset.params(idx_te,5) - ps.dP_list(g)) < 1e-9);
    if numel(mm) < 3, continue; end
    m_g = calc_metrics(y_lab(mm), yhat_all(mm));
    rows = add_row(rows, sprintf('d=%.3gpu', ps.dP_list(g)), '按d分组', m_g);
end
T_group = rows2table(rows);
writetable(T_group, 'results/tables/ta_grouped.csv');

%% ===== 10. 汇总 =====
fprintf('\n===== 主线实验完成,总用时 %.1f 分钟 =====\n', toc(t_start)/60);
fprintf('结果已保存至 results/tables/ta_*.csv\n');
disp(T_main);
end

%% ==================== 本文件的局部函数 ====================

function [PhiB, PhiV] = ta_features(dataset, sim, sys, W, K)
% TA_FEATURES  从截断窗口提取特征(助教规定的三个手工特征 + VMD特征)
% 窗口: [t_dist, t_dist+W]
% 手工特征(无量纲频偏换算成Hz偏差):
%   1) 截止时刻频偏 f_end - f_nom (Hz)
%   2) 窗口平均斜率 mean(diff(f)/diff(t)) (Hz/s)
%   3) 窗口波动幅度 max-min (Hz)
% VMD特征:对窗口偏差(去均值)做VMD(K=3),能量占比×K + 能量熵 = K+1维
N = size(dataset.F, 1);
idx_w = find(dataset.t > sim.t_dist & dataset.t <= sim.t_dist + W);
t_w = dataset.t(idx_w);

PhiB = zeros(N, 3);
PhiV = zeros(N, K+1);
for i = 1:N
    fw = dataset.F(i, idx_w) - sys.f_nom;      % Hz偏差
    PhiB(i, 1) = fw(end);                       % 截止时刻频偏
    PhiB(i, 2) = mean(diff(fw) ./ diff(t_w));   % 平均斜率
    PhiB(i, 3) = max(fw) - min(fw);             % 波动幅度
    % VMD(短窗口可能不稳定,助教允许增益不明显时放弃VMD)
    xw = fw - mean(fw);
    try
        imf = vmd(xw, 'NumIMF', K, 'AbsoluteTolerance', 1e-6);
        if size(imf, 2) == K && size(imf, 1) ~= K, imf = imf.'; end
        e = sum(imf.^2, 2);  p = e / sum(e);  p = max(p, 1e-12);
        PhiV(i, :) = [p.', -sum(p .* log2(p))];
    catch
        PhiV(i, :) = NaN;
    end
    if mod(i, 300) == 0
        fprintf('  特征提取 %d/%d\n', i, N);
    end
end
end

function y = ta_label(dataset, sim, W)
% TA_LABEL  标签 = 窗口截止(t_dist+W)之后的最低点频率
% (与特征窗口自洽:特征用截止前,标签取截止后,无时间重叠 → 无泄露)
t_cut_w = sim.t_dist + W;
N = size(dataset.F, 1);
y = zeros(N, 1);
for i = 1:N
    idx = find(dataset.t > t_cut_w);
    y(i) = min(dataset.F(i, idx));
end
end

function [Z, mu, sg] = zscore_train(Phi, idx_tr)
mu = mean(Phi(idx_tr,:), 1);
sg = std(Phi(idx_tr,:), 1);  sg(sg < 1e-12) = 1;
Z = (Phi - mu) ./ sg;
end

function Z = zscore_apply(Phi, mu, sg)
Z = (Phi - mu) ./ sg;
end

function rows = add_row(rows, name, setting, m)
rows{end+1, 1} = name;
rows{end, 2} = setting;
rows{end, 3} = m.rmse;
rows{end, 4} = m.mae;
rows{end, 5} = m.p95;
rows{end, 6} = m.r2;
rows{end, 7} = m.time_s;
end

function T = rows2table(rows)
T = cell2table(rows, 'VariableNames', ...
    {'Model', 'Setting', 'RMSE_Hz', 'MAE_Hz', 'P95_Hz', 'R2', 'Time_s'});
end

