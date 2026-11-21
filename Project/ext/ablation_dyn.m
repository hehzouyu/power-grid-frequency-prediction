function ablation_dyn()
% ABLATION_DYN  扩展特征消融 [v4:剔除泄露特征depth/rec_rate]  扩展特征消融:仅VMD vs VMD+动态特征(均不给物理参数)
%
% 背景:ablation_vmd v3 的组C(仅4个VMD特征)R²=0.717,说明曲线含参数信息
%       但表达不足。本实验检验:加入5个可解释的物理动态特征后,
%       无参数预测的精度能提升多少。若R²显著上升(如>0.9),
%       "无需参数即可高精度预测Nadar"的论文核心论点即告成立。
%
% 前置: data/processed/features.mat (main阶段1+2)
% 输出: results/tables/ablation_dyn.csv + 条形图

fprintf('[ablation_dyn v1] 开始扩展特征消融实验...\n');

mat_feat = 'data/processed/features.mat';
if ~exist(mat_feat, 'file')
    error('找不到 %s —— 请先运行 main 的阶段1+2。', mat_feat);
end
S = load(mat_feat);
X = S.X;
if isfield(S, 'dataset')
    dataset = S.dataset;
else
    S2 = load('data/raw/dataset_raw.mat');
    dataset = S2.dataset;
end
if size(X,1) ~= size(dataset.F,1)
    error('特征矩阵(%d条)与数据集(%d条)不一致,请重新运行 main 阶段1+2。', ...
        size(X,1), size(dataset.F,1));
end
mc = model_config();
N  = size(X, 1);

% --- 提取动态特征(计算量小,1792条约1分钟) ---
fprintf('提取动态特征(5维)... \n');
D_dyn = zeros(N, 3);
t_vec = dataset.t;
for i = 1:N
    D_dyn(i, :) = dynamic_features(dataset.F(i,:), t_vec, 1.0, 50);
    if mod(i, 400) == 0
        fprintf('  动态特征: %d/%d\n', i, N);
    end
end

% --- 划分(与main一致) ---
rng(mc.seed);
idx = randperm(N);
n_tr = floor(mc.train_ratio * N);
idx_tr = idx(1:n_tr); idx_te = idx(n_tr+1:end);
y_tr = dataset.nadir(idx_tr);
y_te = dataset.nadir(idx_te);

% 组C:仅VMD特征(复现v3组C)
Xc_tr = X(idx_tr, :);  Xc_te = X(idx_te, :);
% 组D:VMD + 动态特征(非泄露3维:RoCoF_max/t_nadir/f_qss)
Xd_tr = [X(idx_tr,:), D_dyn(idx_tr,:)];
Xd_te = [X(idx_te,:), D_dyn(idx_te,:)];
% 参考组A:仅参数(上限参照)
Xa_tr = dataset.params(idx_tr, :);
Xa_te = dataset.params(idx_te, :);

fprintf('训练组C(仅VMD)...      '); tic;
mdl_c = fitrensemble(Xc_tr, y_tr, 'Method', 'LSBoost', ...
    'NumLearningCycles', mc.xgb_cycles, 'LearnRate', mc.xgb_lr); toc;
fprintf('训练组D(VMD+动态)...  '); tic;
mdl_d = fitrensemble(Xd_tr, y_tr, 'Method', 'LSBoost', ...
    'NumLearningCycles', mc.xgb_cycles, 'LearnRate', mc.xgb_lr); toc;
fprintf('训练组A(仅参数,参照)...'); tic;
mdl_a = fitrensemble(Xa_tr, y_tr, 'Method', 'LSBoost', ...
    'NumLearningCycles', mc.xgb_cycles, 'LearnRate', mc.xgb_lr); toc;

m_c = calc_metrics(y_te, predict(mdl_c, Xc_te));
m_d = calc_metrics(y_te, predict(mdl_d, Xd_te));
m_a = calc_metrics(y_te, predict(mdl_a, Xa_te));

T = table({'C_仅VMD特征'; 'D_VMD+动态特征'; 'A_仅参数(上限参照)'}, ...
          [m_c.rmse; m_d.rmse; m_a.rmse], ...
          [m_c.mae;  m_d.mae;  m_a.mae], ...
          [m_c.r2;   m_d.r2;   m_a.r2], ...
    'VariableNames', {'FeatureSet', 'RMSE_Hz', 'MAE_Hz', 'R2'});
disp(T);

fprintf('\n判读:\n');
fprintf('  D的R²比C明显高 -> 动态特征补充了VMD缺失的参数信息,论文核心论点强化\n');
fprintf('  D接近A -> "曲线特征≈参数信息",无需参数的工程愿景成立\n');

if ~exist('results/tables', 'dir'); mkdir('results/tables'); end
writetable(T, 'results/tables/ablation_dyn.csv');

fig = figure('Name', 'ablation_dyn');
bar(categorical(T.FeatureSet), T.R2, 'FaceColor', [0.3 0.75 0.5]);
ylabel('R^2');
title('扩展动态特征的贡献(均无物理参数)');
ylim([0 1.05]);
grid on;
try
    save_results(fig, 'ablation_dyn', T, 'ablation_dyn', 'results');
catch
    exportgraphics(fig, 'results/tables/ablation_dyn.png', 'Resolution', 150);
end
fprintf('结果已保存至 results/tables/ablation_dyn.csv\n');
end
