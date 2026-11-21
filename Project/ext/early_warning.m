function early_warning()
% EARLY_WARNING  早预警统一实验:所有模型只用扰动后前1秒数据预测Nadir
%
% 动机(方法学严谨性):此前LSTM用整条曲线输入,R²=0.9999实质是"从输入里
%   直接读出最低点"(序列包含答案本身),不能算预测;f_qss等末段特征同理。
%   本实验把任务统一为工程上真正有价值的【早预警】场景:
%   故障后仅1秒(100个采样点)内,预测最终的频率最低点Nadir。
%   所有模型用同一窗口、同一划分,对比完全公平。
%
% 特征设计(仅用1秒窗口,杜绝泄露):
%   VMD: 对窗口内偏差信号做VMD(K=3) → 能量占比×3 + 能量熵 = 4维
%   动态: RoCoF_max(窗内最大变化率), df_end(1秒时刻的偏差) = 2维
%   (刻意不使用窗内最小值——极端场景1秒内可能已接近最低点)
%
% 前置: data/raw/dataset_raw.mat (main阶段1)
% 输出: results/tables/early_warning.csv,约15分钟

fprintf('[early_warning v1] 早预警实验:扰动后前1秒 → 预测Nadir\n');
t_start = tic;

S = load('data/raw/dataset_raw.mat');
dataset = S.dataset; sim = S.sim; sys = S.sys;
mc = model_config();
N = size(dataset.F, 1);
K = 3;

% --- 第1步:截取窗口并提取特征 ---
fprintf('步骤1/3: 窗口特征提取(VMD约5-10分钟)...\n');
W = 1.0;                                  % 窗口长度(秒)
idx_w = find(dataset.t >= sim.t_dist & dataset.t <= sim.t_dist + W);
t_w   = dataset.t(idx_w);
X_vmd = zeros(N, K+1);
X_dyn = zeros(N, 2);
for i = 1:N
    fw = dataset.F(i, idx_w);
    fw = fw - mean(fw);                   % 去均值(同vmd_features)
    imf = vmd(fw, 'NumIMF', K, 'AbsoluteTolerance', 1e-6);
    if size(imf, 2) == K && size(imf, 1) ~= K, imf = imf.'; end
    e = sum(imf.^2, 2);  p = e / sum(e);
    p = max(p, 1e-12);
    X_vmd(i, :) = [p.', -sum(p .* log2(p))];
    % 动态特征
    d_fw = diff(fw) ./ diff(t_w);
    X_dyn(i, 1) = max(abs(d_fw));          % RoCoF_max
    X_dyn(i, 2) = fw(end);                 % 1秒时刻偏差(非最小值)
    if mod(i, 300) == 0
        fprintf('  %d/%d (%.0f%%)\n', i, N, 100*i/N);
    end
end
X_early = [X_vmd, X_dyn];                  % 6维早预警特征
fprintf('特征提取完成,用时 %.1f 分钟\n', toc(t_start)/60);

% --- 第2步:统一划分(与main一致) ---
rng(mc.seed);
idx = randperm(N);
n_tr = floor(mc.train_ratio * N);
idx_tr = idx(1:n_tr); idx_te = idx(n_tr+1:end);
y_tr = dataset.nadir(idx_tr); y_te = dataset.nadir(idx_te);

% LSTM输入:窗口原始曲线(偏差形式),记录训练集归一化参数
mu_f = mean(dataset.F(idx_tr, idx_w), 'all');
sg_f = std(dataset.F(idx_tr, idx_w), 0, 'all');
F_tr_n = (dataset.F(idx_tr, idx_w) - mu_f) / sg_f;
F_te_n = (dataset.F(idx_te, idx_w) - mu_f) / sg_f;
mu_y = mean(y_tr); sg_y = std(y_tr);
y_tr_n = (y_tr - mu_y) / sg_y;

% --- 第3步:训练四个模型 ---
fprintf('步骤2/3: 训练模型...\n');
% 解析法基线
yhat_ana = baseline_analytic(dataset.params(idx_te, :), sys);
% 线性回归
mdl_lin = fitlm(X_early(idx_tr, :), y_tr);
yhat_lin = predict(mdl_lin, X_early(idx_te, :));
% XGBoost
mdl_xgb = fitrensemble(X_early(idx_tr, :), y_tr, 'Method', 'LSBoost', ...
    'NumLearningCycles', mc.xgb_cycles, 'LearnRate', mc.xgb_lr);
yhat_xgb = predict(mdl_xgb, X_early(idx_te, :));
% LSTM(输入=1秒窗口,目标归一化,同train_models的修复经验)
fprintf('  LSTM训练中(约5-10分钟)...\n');
layers = [ sequenceInputLayer(1)
           lstmLayer(mc.lstm_hidden, 'OutputMode', 'last')
           fullyConnectedLayer(1)
           regressionLayer ];
opts = trainingOptions('adam', 'MaxEpochs', mc.lstm_epochs, ...
    'GradientThreshold', 1, 'Verbose', false, 'Plots', 'none');
lstm_net = trainNetwork(num2cell(F_tr_n, 2), y_tr_n, layers, opts);
yhat_lstm = predict(lstm_net, num2cell(F_te_n, 2)) * sg_y + mu_y;

% --- 第4步:评估输出 ---
fprintf('步骤3/3: 评估...\n');
names = {'解析法基线'; '线性回归'; 'XGBoost'; 'LSTM'};
yhats = {yhat_ana; yhat_lin; yhat_xgb; yhat_lstm};
rmse_v = zeros(4,1); mae_v = zeros(4,1); r2_v = zeros(4,1);
for k = 1:4
    m = calc_metrics(y_te, yhats{k});
    rmse_v(k) = m.rmse; mae_v(k) = m.mae; r2_v(k) = m.r2;
end
T = table(names, rmse_v, mae_v, r2_v, ...
    'VariableNames', {'Model', 'RMSE_Hz', 'MAE_Hz', 'R2'});
disp(T);

if ~exist('results/tables', 'dir'); mkdir('results/tables'); end
writetable(T, 'results/tables/early_warning.csv');

fig = figure('Name', 'early_warning');
bar(categorical(T.Model), T.R2, 'FaceColor', [0.35 0.65 0.9]);
ylabel('R^2'); ylim([0 1.05]);
title('早预警任务:扰动后1秒预测Nadir');
grid on;
try
    save_results(fig, 'early_warning', T, 'early_warning', 'results');
catch
    exportgraphics(fig, 'results/tables/early_warning.png', 'Resolution', 150);
end
fprintf('完成,总用时 %.1f 分钟,结果已保存\n', toc(t_start)/60);
end
