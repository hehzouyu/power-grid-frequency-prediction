function ablation_vmd()
% ABLATION_VMD  消融实验:验证VMD特征对预测精度的真实贡献 [版本v3 三组对比]
%
% 动机(大白话):XGBoost用"参数+VMD特征"得到了R²=0.985,但Pearson筛选显示
%        每个VMD特征和目标的相关性都很弱——那这0.985到底谁挣来的?
%        消融实验的做法是"做减法":把VMD特征拿掉,只留5个原始参数再训一遍,
%        比较两者精度,判断VMD特征是否提供了独立信息。
%
% 前置:需先有 data/processed/features.mat(运行 main 阶段1+2 生成)
%
% 输出:命令行对比表 + results/tables/ablation_vmd.csv + 条形图

fprintf('[ablation_vmd v3] 开始消融实验(三组对比)...\n');

% --- 显式加载(不用裸load,避免任何变量未注入的歧义) ---
mat_feat = 'data/processed/features.mat';
if ~exist(mat_feat, 'file')
    error('找不到 %s —— 请先运行 main 的阶段1+2 生成数据。', mat_feat);
end
S = load(mat_feat);
X = S.X;
if isfield(S, 'dataset')
    dataset = S.dataset;
else
    fprintf('features.mat中无dataset变量,改从data/raw/dataset_raw.mat加载\n');
    S2 = load('data/raw/dataset_raw.mat');
    dataset = S2.dataset;
end
% 一致性检查:特征矩阵与数据集条数必须一致
if size(X, 1) ~= size(dataset.F, 1)
    error('特征矩阵(%d条)与数据集(%d条)不一致——features.mat是旧版,请重新运行 main 阶段1+2。', ...
        size(X,1), size(dataset.F,1));
end
fprintf('数据加载完成: %d 条曲线,特征 %d 维\n', size(X,1), size(X,2));

mc = model_config();
N  = size(X, 1);

% --- 与main一致的划分 ---
rng(mc.seed);
idx = randperm(N);
n_tr = floor(mc.train_ratio * N);
idx_tr = idx(1:n_tr); idx_te = idx(n_tr+1:end);

y_tr = dataset.nadir(idx_tr);
y_te = dataset.nadir(idx_te);

% --- 三组对比实验 ---
% 组A:只用传统参数(H, D, Km, dP, pen) —— 参数已知的理想情况
Xa_tr = dataset.params(idx_tr, :);
Xa_te = dataset.params(idx_te, :);

% 组B:传统参数 + VMD特征(与main正式流程一致)
Xb_tr = [X(idx_tr, :), dataset.params(idx_tr, :)];
Xb_te = [X(idx_te, :),   dataset.params(idx_te, :)];

% 组C(最重要):只用VMD特征,不给任何参数 —— 对应工程实际中参数未知/时变的场景,
%           也是本课题的立论出发点:"无需知道系统内部参数,仅从频率曲线预测Nadir"
Xc_tr = X(idx_tr, :);
Xc_te = X(idx_te, :);

fprintf('训练组A(仅参数)...\n');
mdl_a = fitrensemble(Xa_tr, y_tr, 'Method', 'LSBoost', ...
    'NumLearningCycles', mc.xgb_cycles, 'LearnRate', mc.xgb_lr);
fprintf('训练组B(参数+VMD)...\n');
mdl_b = fitrensemble(Xb_tr, y_tr, 'Method', 'LSBoost', ...
    'NumLearningCycles', mc.xgb_cycles, 'LearnRate', mc.xgb_lr);
fprintf('训练组C(仅VMD特征,无参数)...\n');
mdl_c = fitrensemble(Xc_tr, y_tr, 'Method', 'LSBoost', ...
    'NumLearningCycles', mc.xgb_cycles, 'LearnRate', mc.xgb_lr);

m_a = calc_metrics(y_te, predict(mdl_a, Xa_te));
m_b = calc_metrics(y_te, predict(mdl_b, Xb_te));
m_c = calc_metrics(y_te, predict(mdl_c, Xc_te));

% --- 输出 ---
T = table({'A_仅传统参数'; 'B_参数+VMD特征'; 'C_仅VMD特征(无参数)'}, ...
          [m_a.rmse; m_b.rmse; m_c.rmse], ...
          [m_a.mae;  m_b.mae;  m_c.mae], ...
          [m_a.r2;   m_b.r2;   m_c.r2], ...
    'VariableNames', {'FeatureSet', 'RMSE_Hz', 'MAE_Hz', 'R2'});
disp(T);

fprintf('\n结论判读:\n');
fprintf('  A≈B(已验证):参数已知时VMD特征无额外贡献,符合"参数是充分信息"的物理直觉\n');
fprintf('  C是关键:若C的R²仍较高(如>0.9) -> "无需参数即可预测"成立,这是论文核心卖点\n');
fprintf('  若C明显弱于A -> 曲线特征携带参数信息有限,如实讨论,可结合扩展特征改进\n');

% --- 保存 ---
if ~exist('results/tables', 'dir'); mkdir('results/tables'); end
writetable(T, 'results/tables/ablation_vmd.csv');

fig = figure('Name', 'ablation_vmd');
bar(categorical(T.FeatureSet), T.RMSE_Hz, 'FaceColor', [0.2 0.6 0.9]);
ylabel('RMSE (Hz)');
title('消融实验:VMD特征的贡献');
grid on;
try
    save_results(fig, 'ablation_vmd', T, 'ablation_vmd', 'results');
catch
    % save_results 不在路径上时直接保存PNG
    exportgraphics(fig, 'results/tables/ablation_vmd.png', 'Resolution', 150);
end
fprintf('结果已保存至 results/tables/ablation_vmd.csv\n');
end
