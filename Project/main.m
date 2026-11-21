%% MAIN  总控脚本:一键运行完整流程
% 大白话:这是整个工程的"总开关面板"。
%        每个阶段对应一个布尔开关,做完一个阶段就把它关掉(或开着复跑),
%        后面的阶段依赖前面的数据文件,顺序不要乱。
%
% 阶段对应关系:
%   stage1  物理仿真(生成数据集)        对应第1~2周
%   stage2  VMD特征提取 + Pearson筛选    对应第3周
%   stage3  模型训练 + 指标对比          对应第4周
%   stage4  噪声鲁棒性 + 热-电耦合验证   对应第5周

clear; clc; close all;
addpath(genpath(pwd));          % 把整个工程加进路径

%% 配置加载
sys = sys_config();
sim = sim_config();
ps  = param_space('full');   % 正式实验用1792组;测试脚本仍用默认demo模式
mc  = model_config();

%% 阶段开关(true = 执行, false = 跳过)
run.stage1 = true;      % 批量仿真生成数据集
run.stage2 = true;      % VMD特征提取与筛选
run.stage3 = true;      % 模型训练与对比
run.stage4 = true;      % 扩展场景鲁棒性验证

%% ===== 阶段1:物理仿真与数据集 =====
if run.stage1
    fprintf('\n========== 阶段1:批量仿真 ==========\n');
    dataset = gen_dataset(sys, sim, ps);
    if ~exist('data/raw', 'dir'); mkdir('data/raw'); end
    save('data/raw/dataset_raw.mat', 'dataset', 'sys', 'sim', 'ps');
else
    load('data/raw/dataset_raw.mat');
end

%% ===== 阶段2:VMD特征提取与筛选 =====
if run.stage2
    fprintf('\n========== 阶段2:VMD特征提取 ==========\n');
    K = 3;
    N = size(dataset.F, 1);
    X = zeros(N, K+1);
    for i = 1:N
        X(i, :) = vmd_features(dataset.F(i, :), K).vec;
        if mod(i, 100) == 0
            fprintf('  VMD特征提取: %d/%d (%.0f%%)\n', i, N, 100*i/N);
        end
    end
    % 动态物理特征(RoCoF_max/t_nadir/f_qss,非泄露3维,约1分钟)
    % 注意:早期版本含depth/rec_rate两列,与标签Nadar恒等(特征泄露),
    %       已剔除——线性回归出现R²=1.0即泄露的铁证
    fprintf('提取动态特征(3维,非泄露)...\n');
    D_dyn = zeros(N, 3);
    for i = 1:N
        D_dyn(i, :) = dynamic_features(dataset.F(i,:), dataset.t, sim.t_dist, sys.f_nom);
    end
    % 官方特征:VMD + 动态特征(不给物理参数,对应论文核心方法)
    X_official = [X, D_dyn];
    [r_sorted, sel_idx] = pearson_select(X, dataset.nadir, 0.3);
    fprintf('Pearson筛选: %d/%d 个特征入选\n', numel(sel_idx), K+1);
    fprintf('各特征|r|排序:\n');
    for j = 1:size(r_sorted,1)
        fprintf('  特征%d: |r| = %.3f\n', r_sorted(j,1), r_sorted(j,2));
    end
    if isempty(sel_idx)
        fprintf('提示:无特征过阈值——VMD特征可能与Nadir非线性相关,\n');
        fprintf('      可运行 ext/ablation_vmd.m 用消融实验检验其实际贡献。\n');
    end
    if ~exist('data/processed', 'dir'); mkdir('data/processed'); end
    save('data/processed/features.mat', 'X', 'D_dyn', 'X_official', 'sel_idx', 'K', 'dataset');
else
    load('data/processed/features.mat');
end

%% ===== 阶段3:模型训练与对比 =====
if run.stage3
    fprintf('\n========== 阶段3:模型训练 ==========\n');
    % 官方方法:X_official = VMD+动态特征(无参数);
    % 另保留"参数+VMD"对照组,用于展示参数已知时的上限
    if ~exist('X_official', 'var')
        load('data/processed/features.mat', 'X_official');
    end
    % 防泄露:旧版X_official=[X(4列),D_dyn(5列)]共9列,含depth/rec_rate泄露列,
    % 剔除后保留 [X, RoCoF_max, t_nadir, f_qss] 共7列
    if size(X_official, 2) == 9
        X_official = X_official(:, [1:4, 5, 6, 8]);
        fprintf('已从旧版X_official中剔除泄露特征(depth/rec_rate)\n');
    end
    X_aug  = X_official;                 % 官方特征(无参数)
    X_augP = [X, dataset.params];        % 参数对照组
    rng(mc.seed);
    N = size(X_aug, 1);
    idx = randperm(N);
    n_tr = floor(mc.train_ratio * N);
    idx_tr = idx(1:n_tr); idx_te = idx(n_tr+1:end);

    models = train_models(X_aug(idx_tr,:), dataset.nadir(idx_tr), ...
                          dataset.F(idx_tr,:), mc);
    results = eval_models(models, X_aug(idx_te,:), dataset.nadir(idx_te), ...
                          dataset.F(idx_te,:), dataset.params(idx_te,:), sys);
    % 参数对照组:XGBoost 用"参数+VMD"特征再训一遍,比较官方方法的差距
    mdl_P = fitrensemble(X_augP(idx_tr,:), dataset.nadir(idx_tr), 'Method', 'LSBoost', ...
        'NumLearningCycles', mc.xgb_cycles, 'LearnRate', mc.xgb_lr);
    mP = calc_metrics(dataset.nadir(idx_te), predict(mdl_P, X_augP(idx_te,:)));
    fprintf('对照:XGBoost(参数+VMD) RMSE = %.5f Hz, R2 = %.4f\n', mP.rmse, mP.r2);
    if ~exist('results/tables', 'dir'); mkdir('results/tables'); end
    disp(results(:, 1:4));
    save('results/tables/model_comparison.mat', 'results', 'models', 'idx_tr', 'idx_te');
    writetable(results(:, 1:4), 'results/tables/model_comparison.csv');
    fprintf('模型对比表已导出: results/tables/model_comparison.csv\n');
end

%% ===== 阶段4:扩展场景鲁棒性验证 =====
if ~exist('results/figures', 'dir'); mkdir('results/figures'); end
if run.stage4
    fprintf('\n========== 阶段4:鲁棒性验证 ==========\n');

    % 4a. 热-电耦合:虚拟惯量对Nadir的改善
    [~, ~, nadir_plain] = simulate_freq(sys, sim);
    [~, ~, nadir_heat ] = heat_coupling(sys, sim, mc.H_vhp);
    fprintf('虚拟惯量 %.1f s: Nadir %.4f → %.4f Hz (改善 %.4f)\n', ...
        mc.H_vhp, nadir_plain, nadir_heat, nadir_heat - nadir_plain);

    % 4b. 噪声鲁棒性:调用测试5的核心逻辑(此处略,运行 test_noise 查看完整流程)
    fprintf('噪声鲁棒性:请运行 tests/test_noise.m 查看 SNR-RMSE 曲线\n');
end

fprintf('\n全部阶段执行完毕。\n');
