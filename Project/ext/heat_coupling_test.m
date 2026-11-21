function heat_coupling_test()
% HEAT_COUPLING_TEST  热-电耦合场景下的模型泛化能力测试
%
% 大白话:热泵提供虚拟惯量后,电网的动态特性变了(跌落更浅)。
%        问题是:在"没有热泵"的数据上训练的模型,遇到热泵场景还行不行?
%        做法:抽300个场景,分别加不同虚拟惯量H_vhp重新仿真,
%        提取同样的特征,用原模型预测,看RMSE随H_vhp的变化。
%        这检验的是模型的【泛化/迁移能力】——热-电耦合是训练时没见过的场景。
%
% 前置: data/processed/features.mat; simulate_freq / dynamic_features / vmd_features
% 输出: results/tables/heat_coupling_test.csv + 曲线图

fprintf('[heat_coupling_test] 开始...\n');
S = load('data/processed/features.mat');
dataset = S.dataset;
sys = sys_config(); sim = sim_config(); mc = model_config();
X_off = S.X;
% 旧版features.mat可能缺D_dyn:现场补算动态特征(约1分钟)
if ~isfield(S, 'D_dyn')
    fprintf('features.mat无D_dyn,现场补算动态特征...\n');
    Dd = zeros(size(X_off,1), 3);
    for i = 1:size(X_off,1)
        Dd(i,:) = dynamic_features(dataset.F(i,:), dataset.t, sim.t_dist, sys.f_nom);
    end
    X_off = [X_off, Dd];
else
    D_dyn = S.D_dyn;
% 兼容与防泄露:旧版D_dyn含5列,其中depth/rec_rate与标签恒等(泄露),
% 统一剔除只保留[RoCoF_max, t_nadir, f_qss]三列
if size(D_dyn, 2) == 5
    D_dyn = D_dyn(:, [1 2 4]);
end
    X_off = [X_off, D_dyn];
end
N = size(X_off, 1);

% 用官方特征训练基准模型(与main一致的训练集)
rng(mc.seed);
idx = randperm(N);
n_tr = floor(mc.train_ratio * N);
mdl = fitrensemble(X_off(idx(1:n_tr),:), dataset.nadir(idx(1:n_tr)), ...
    'Method', 'LSBoost', 'NumLearningCycles', mc.xgb_cycles, 'LearnRate', mc.xgb_lr);
fprintf('基准模型训练完成\n');

% 抽样300个场景做热耦合测试
rng(42);
sub = sort(randperm(N, 300));

H_vhp_list = [0, 1, 2, 4];
rmse_h = zeros(numel(H_vhp_list),1);
nadirs_true = cell(numel(H_vhp_list),1); nadirs_pred = cell(numel(H_vhp_list),1);

for h = 1:numel(H_vhp_list)
    Hv = H_vhp_list(h);
    F_h = zeros(300, numel(dataset.t));
    y_h = zeros(300, 1);
    for i = 1:300
        p = dataset.params(sub(i), :);
        s = sys;
        s.H = p(1);  s.D = p(2);  s.K_m = p(3);   % 该场景的等效参数
        sm = sim;    sm.dP = p(4);
        s.H = s.H + Hv;                            % 叠加热泵虚拟惯量
        [~, f_h, y_h(i)] = simulate_freq(s, sm);
        F_h(i, :) = f_h;
    end
    % 提取同样的官方特征
    X_h = zeros(300, size(X_off,2));
    for i = 1:300
        fv = vmd_features(F_h(i,:), 3).vec;
        fd = dynamic_features(F_h(i,:), dataset.t, sim.t_dist, sys.f_nom);
        X_h(i,:) = [fv, fd];
    end
    yhat = predict(mdl, X_h);
    met = calc_metrics(y_h, yhat);
    rmse_h(h) = met.rmse;
    nadirs_true{h} = y_h; nadirs_pred{h} = yhat;
    fprintf('H_vhp = %d s: RMSE = %.5f Hz\n', Hv, met.rmse);
end

T = table(H_vhp_list(:), rmse_h, 'VariableNames', {'H_vhp_s', 'RMSE_Hz'});
disp(T);
writetable(T, 'results/tables/heat_coupling_test.csv');

fig = figure('Name', 'heat_coupling_test');
plot(H_vhp_list, rmse_h*1000, '-o', 'LineWidth', 1.5, ...
    'MarkerFaceColor', [0.85 0.5 0.2], 'MarkerEdgeColor', 'k');
xlabel('热泵虚拟惯量 H_{vhp} (s)');
ylabel('RMSE (mHz)');
title('热-电耦合场景下的模型泛化能力');
grid on;
try
    save_results(fig, 'heat_coupling_test', T, 'heat_coupling_test', 'results');
catch
    exportgraphics(fig, 'results/tables/heat_coupling_test.png', 'Resolution', 150);
end
fprintf('结果已保存至 results/tables/heat_coupling_test.csv\n');
end
