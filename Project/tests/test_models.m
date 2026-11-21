%% TEST_MODELS  测试4:验证三个模型训练与指标对比流程
% 测试目标:
%   1) 数据划分正确(70%训练 / 30%测试)
%   2) 三个模型都能训练成功,预测无NaN
%   3) 指标对比表能生成,且数据驱动模型应优于解析法基线
%   4) 预测散点图:点应大致落在 y=x 对角线附近
%
% 通过标准:断言全部通过 + 指标表数值合理

clear; clc; close all;
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
projRoot = fileparts(fileparts(mfilename('fullpath')));  % 工程根目录(绝对路径,不受运行位置影响)

%% 第1步:准备数据(特征 = VMD特征 融合 传统参数)
if exist(fullfile(projRoot,'data/processed/features.mat'), 'file')
    load(fullfile(projRoot,'data/processed/features.mat'));
else
    sys = sys_config(); sim = sim_config(); ps = param_space();
    dataset = gen_dataset(sys, sim, ps);
    K = 3; N = size(dataset.F, 1);
    X_vmd = zeros(N, K+1);
    for i = 1:N
        X_vmd(i, :) = vmd_features(dataset.F(i, :), K).vec;
    end
    X = X_vmd;  dataset_ = dataset; %#ok<NASGU>
    save(fullfile(projRoot,'data/processed/features.mat'), 'X', 'K', 'dataset');
end
sys = sys_config();
mc  = model_config();

N = size(X, 1);
% 融合:VMD特征 + 传统参数(H等效值、D、Km、dP、pen),共 (K+1)+5 维
X_aug = [X, dataset.params];

%% 第2步:划分训练/测试集(固定随机种子,可复现)
rng(mc.seed);
idx = randperm(N);
n_tr = floor(mc.train_ratio * N);
idx_tr = idx(1:n_tr); idx_te = idx(n_tr+1:end);

X_tr = X_aug(idx_tr, :);  y_tr = dataset.nadir(idx_tr);
X_te = X_aug(idx_te, :);  y_te = dataset.nadir(idx_te);
F_tr = dataset.F(idx_tr, :); F_te = dataset.F(idx_te, :);
params_te = dataset.params(idx_te, :);

fprintf('数据划分: %d 训练 / %d 测试\n', numel(idx_tr), numel(idx_te));

%% 第3步:训练
models = train_models(X_tr, y_tr, F_tr, mc);
fprintf('[通过] 三个模型训练完成\n');

%% 第4步:评估
results = eval_models(models, X_te, y_te, F_te, params_te, sys);
disp(results(:, 1:4));    % 打印指标表

% 断言1:所有预测无NaN
yhat_xgb = results.yhat{3};
assert(~any(isnan(yhat_xgb)), 'XGBoost预测存在NaN!');
assert(~any(isnan(results.yhat{4})), 'LSTM预测存在NaN!');
fprintf('[通过] 预测无NaN\n');

% 断言2:XGBoost应优于解析法基线(数据驱动的核心论点)
assert(results.RMSE_Hz(3) < results.RMSE_Hz(1), ...
    'XGBoost未超过解析法基线,需检查特征或数据!');
fprintf('[通过] XGBoost RMSE (%.4f) < 解析法 (%.4f)\n', ...
    results.RMSE_Hz(3), results.RMSE_Hz(1));

%% 第5步:预测散点图(XGBoost)
figure('Name', 'test_models');
plot(y_te, y_te, '--k', 'LineWidth', 1); hold on;      % 完美预测对角线
plot(y_te, yhat_xgb, 'o', 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.2 0.6 0.9], 'MarkerEdgeColor', 'k');
xlabel('真实 Nadir (Hz)'); ylabel('预测 Nadir (Hz)');
title(sprintf('XGBoost预测 vs 真实 (RMSE = %.4f Hz)', results.RMSE_Hz(3)));
legend('理想预测', 'XGBoost', 'Location', 'southeast');
grid on;

%% 第6步:保存结果
save_results(gcf, 'test_models_scatter', results(:, 1:4), 'model_comparison', fullfile(projRoot,'results'));
fprintf('[通过] 测试4完成,结果已保存\n');
