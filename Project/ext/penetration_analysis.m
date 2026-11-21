function penetration_analysis()
% PENETRATION_ANALYSIS  分析新能源渗透率对模型预测精度的影响
%
% 大白话:渗透率越高,电网惯量越低,频率动态越"恶劣"。
%        我们关心:模型在渗透率高的场景下会不会变差?
%        做法:按渗透率把测试集分组,分别算R²,画"渗透率-精度"曲线。
%
% 前置: data/processed/features.mat (main阶段1+2,需含D_dyn)
% 输出: results/tables/penetration_analysis.csv + 曲线图

fprintf('[penetration_analysis] 开始...\n');
S = load('data/processed/features.mat');
dataset = S.dataset;
X = S.X;
% 旧版features.mat可能缺D_dyn:现场补算动态特征(约1分钟)
if ~isfield(S, 'D_dyn')
    fprintf('features.mat无D_dyn,现场补算动态特征...\n');
    sys0 = sys_config(); sim0 = sim_config();
    Dd = zeros(size(X,1), 3);
    for i = 1:size(X,1)
        Dd(i,:) = dynamic_features(dataset.F(i,:), dataset.t, sim0.t_dist, sys0.f_nom);
    end
    X = [X, Dd];
else
    D_dyn = S.D_dyn;
% 兼容与防泄露:旧版D_dyn含5列,其中depth/rec_rate与标签恒等(泄露),
% 统一剔除只保留[RoCoF_max, t_nadir, f_qss]三列
if size(D_dyn, 2) == 5
    D_dyn = D_dyn(:, [1 2 4]);
end
    X = [X, D_dyn];
end
mc = model_config();
N = size(X, 1);

rng(mc.seed);
idx = randperm(N);
n_tr = floor(mc.train_ratio * N);
idx_tr = idx(1:n_tr); idx_te = idx(n_tr+1:end);

mdl = fitrensemble(X(idx_tr,:), dataset.nadir(idx_tr), 'Method', 'LSBoost', ...
    'NumLearningCycles', mc.xgb_cycles, 'LearnRate', mc.xgb_lr);
yhat = predict(mdl, X(idx_te,:));

% 按渗透率分组评估
pen_te = dataset.params(idx_te, 5);
pen_levels = unique(pen_te);
n_lvl = numel(pen_levels);
rmse_lvl = zeros(n_lvl,1); r2_lvl = zeros(n_lvl,1); count_lvl = zeros(n_lvl,1);
for j = 1:n_lvl
    m = pen_te == pen_levels(j);
    met = calc_metrics(dataset.nadir(idx_te(m)), yhat(m));
    rmse_lvl(j) = met.rmse; r2_lvl(j) = met.r2; count_lvl(j) = sum(m);
end

T = table(pen_levels, count_lvl, rmse_lvl, r2_lvl, ...
    'VariableNames', {'Penetration', 'N_test', 'RMSE_Hz', 'R2'});
disp(T);
writetable(T, 'results/tables/penetration_analysis.csv');

fig = figure('Name', 'penetration_analysis');
yyaxis left
plot(pen_levels*100, rmse_lvl*1000, '-o', 'LineWidth', 1.5);
ylabel('RMSE (mHz)');
yyaxis right
plot(pen_levels*100, r2_lvl, '--s', 'LineWidth', 1.5);
ylabel('R^2');
xlabel('新能源渗透率 (%)');
title('模型精度随新能源渗透率的变化');
grid on;
try
    save_results(fig, 'penetration_analysis', T, 'penetration_analysis', 'results');
catch
    exportgraphics(fig, 'results/tables/penetration_analysis.png', 'Resolution', 150);
end
fprintf('结果已保存至 results/tables/penetration_analysis.csv\n');
end
