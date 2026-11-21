function early_warning_sweep()
% EARLY_WARNING_SWEEP  早预警窗口长度扫描:W = 0.5/1/2/3 秒
% 输出: results/tables/window_sweep.csv (Window_s × Model 的 RMSE/MAE/R2)
% 预计总耗时约50分钟(VMD特征提取为大头)

fprintf('[early_warning_sweep] 窗口长度扫描开始...\n');
t0 = tic;

S = load('data/raw/dataset_raw.mat');
dataset = S.dataset; sim = S.sim; sys = S.sys;
mc = model_config();
N = size(dataset.F, 1);
K = 3;

% 统一划分(与 early_warning 一致)
rng(mc.seed);
idx = randperm(N);
n_tr = floor(mc.train_ratio * N);
idx_tr = idx(1:n_tr); idx_te = idx(n_tr+1:end);
y_tr = dataset.nadir(idx_tr); y_te = dataset.nadir(idx_te);

W_list = [0.5, 1.0, 2.0, 3.0];
model_names = {'解析法基线', '线性回归', 'XGBoost', 'LSTM'};

% 预存LSTM归一化参数随窗口变化,逐窗口计算
results = [];

for wi = 1:numel(W_list)
    W = W_list(wi);
    fprintf('\n===== 窗口 W = %.1f s (%d/%d) =====\n', W, wi, numel(W_list));

    % --- 第1步:窗口特征 ---
    idx_w = find(dataset.t >= sim.t_dist & dataset.t <= sim.t_dist + W);
    t_w = dataset.t(idx_w);
    X_vmd = zeros(N, K+1); X_dyn = zeros(N, 2);
    for i = 1:N
        fw = dataset.F(i, idx_w); fw = fw - mean(fw);
        imf = vmd(fw, 'NumIMF', K, 'AbsoluteTolerance', 1e-6);
        if size(imf, 2) == K && size(imf, 1) ~= K, imf = imf.'; end
        e = sum(imf.^2, 2); p = e/sum(e); p = max(p, 1e-12);
        X_vmd(i, :) = [p.', -sum(p .* log2(p))];
        X_dyn(i, 1) = max(abs(diff(fw) ./ diff(t_w)));
        X_dyn(i, 2) = fw(end);
        if mod(i, 400) == 0
            fprintf('  特征 %d/%d\n', i, N);
        end
    end
    X_early = [X_vmd, X_dyn];

    % --- 第2步:训练四个模型 ---
    yhat_ana = baseline_analytic(dataset.params(idx_te, :), sys);
    mdl_lin = fitlm(X_early(idx_tr, :), y_tr);
    yhat_lin = predict(mdl_lin, X_early(idx_te, :));
    mdl_xgb = fitrensemble(X_early(idx_tr, :), y_tr, 'Method', 'LSBoost', ...
        'NumLearningCycles', mc.xgb_cycles, 'LearnRate', mc.xgb_lr);
    yhat_xgb = predict(mdl_xgb, X_early(idx_te, :));

    % LSTM(窗口数据归一化)
    mu_f = mean(dataset.F(idx_tr, idx_w), 'all');
    sg_f = std(dataset.F(idx_tr, idx_w), 0, 'all');
    mu_y = mean(y_tr); sg_y = std(y_tr);
    F_tr_n = (dataset.F(idx_tr, idx_w) - mu_f) / sg_f;
    F_te_n = (dataset.F(idx_te, idx_w) - mu_f) / sg_f;
    layers = [ sequenceInputLayer(1)
               lstmLayer(mc.lstm_hidden, 'OutputMode', 'last')
               fullyConnectedLayer(1)
               regressionLayer ];
    opts = trainingOptions('adam', 'MaxEpochs', mc.lstm_epochs, ...
        'GradientThreshold', 1, 'Verbose', false, 'Plots', 'none');
    net = trainNetwork(num2cell(F_tr_n, 2), (y_tr - mu_y)/sg_y, layers, opts);
    yhat_lstm = predict(net, num2cell(F_te_n, 2)) * sg_y + mu_y;

    % --- 第3步:评估记录 ---
    yhats = {yhat_ana, yhat_lin, yhat_xgb, yhat_lstm};
    for m = 1:4
        met = calc_metrics(y_te, yhats{m});
        fprintf('  %s: RMSE=%.5f Hz, R2=%.4f\n', model_names{m}, met.rmse, met.r2);
        results = [results; W, m, met.rmse, met.mae, met.r2]; %#ok<AGROW>
    end
end

T = array2table(results, 'VariableNames', ...
    {'Window_s', 'ModelID', 'RMSE_Hz', 'MAE_Hz', 'R2'});
T.Model = model_names(T.ModelID).';
if ~exist('results/tables', 'dir'), mkdir('results/tables'); end
writetable(T, 'results/tables/window_sweep.csv');
fprintf('\n全部完成,用时 %.1f 分钟,已保存 window_sweep.csv\n', toc(t0)/60);
fprintf('请把 results/tables/window_sweep.csv 内容发给助手画图\n');
end