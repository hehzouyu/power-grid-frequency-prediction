function make_ta_figures()
% MAKE_TA_FIGURES  生成助教版论文全部插图(2026-09-24)
% 图内无标题(标题由LaTeX caption统一编号),PNG 300dpi
% 输出: ./ta_figs/  (用户自行把PNG复制到论文的 figures/ 目录)
%
% 前置: data/raw/dataset_ta.mat 与 results/tables/ta_*.csv 已存在
%       (先运行 main_ta_pipeline)

outdir = 'ta_figs';
if ~exist(outdir, 'dir'), mkdir(outdir); end

sys = sys_config();  sim = sim_config();
mat_ta = 'data/raw/dataset_ta.mat';
if ~exist(mat_ta, 'file')
    error('找不到 %s —— 请先运行 main_ta_pipeline 生成数据集。', mat_ta);
end
S = load(mat_ta);
if ~isfield(S, 'dataset')
    error('%%s 中没有 dataset 变量,文件可能损坏或是旧格式,请删除后重跑 main_ta_pipeline。', mat_ta);
end
dataset = S.dataset;
need = {'F','nadir','params','posthoc','t'};
miss = need(~ismember(need, fieldnames(dataset)));
if ~isempty(miss)
    error('dataset 缺少字段: %s —— 数据集是旧格式,请删除 %s 后重跑 main_ta_pipeline。', ...
        strjoin(miss, ', '), mat_ta);
end
t_cut = sim.t_dist + sim.tau_obs;

%% ===== 图1:时间轴示意(可见0.5s → 预测后续最低点) =====
% 选一条典型曲线(中等参数)
[~, i_mid] = min(abs(dataset.params(:,1)-4) + abs(dataset.params(:,5)-0.05));
t = dataset.t;  f = dataset.F(i_mid, :);
fig1 = figure('Color','w','Position',[100 100 800 420]);
plot(t, f, 'b-', 'LineWidth', 1.6); hold on;
% 可见窗口(0~截止)与不可见区底色区分
patch([t(1) t_cut t_cut t(1)], [49.85 49.85 50.02 50.02], ...
    [0.85 0.95 0.85], 'EdgeColor','none', 'FaceAlpha', 0.5);
plot(t, f, 'b-', 'LineWidth', 1.6);
xline(sim.t_dist, '--r', 'LineWidth', 1.2);
xline(t_cut, '-', 'Color', [0 0.5 0], 'LineWidth', 1.8);
% 标签最低点(截止后)
idx_after = find(t > t_cut);
[fnad, k] = min(f(idx_after));
tnad = t(idx_after(k));
plot(tnad, fnad, 'rp', 'MarkerSize', 14, 'MarkerFaceColor', 'r');
text(tnad+0.3, fnad, sprintf('标签最低点\n%.4f Hz', fnad), 'Color','r', 'FontSize', 10);
text(0.4, 49.99, {'可见窗口(0.5s)','特征仅取自这里'}, 'FontSize', 10, 'Color', [0 0.4 0]);
text(3.0, 49.965, {'不可见区:预测目标所在','标签=截止后最低点'}, 'FontSize', 10, 'Color', [0.4 0 0]);
xlabel('时间 (s)'); ylabel('频率 (Hz)');
xlim([0 8]); ylim([49.85 50.02]);
grid on; set(gca, 'FontSize', 11);
exportgraphics(fig1, fullfile(outdir, 'ta_fig1_timeline.png'), 'Resolution', 300);
close(fig1);

%% ===== 图2:数据集曲线簇 =====
fig2 = figure('Color','w','Position',[100 100 800 450]);
hold on;
idx_show = round(linspace(1, size(dataset.F,1), 12));
for k = idx_show
    plot(t, dataset.F(k,:), 'LineWidth', 0.9);
end
xline(sim.t_dist, '--r', 'LineWidth', 1);
xline(t_cut, '-', 'Color', [0 0.5 0], 'LineWidth', 1.4);
xlabel('时间 (s)'); ylabel('频率 (Hz)');
xlim([0 10]);
grid on; set(gca, 'FontSize', 11);
exportgraphics(fig2, fullfile(outdir, 'ta_fig2_dataset.png'), 'Resolution', 300);
close(fig2);

%% ===== 图3:主结果(四模型R²对比) =====
T_main = readtable('results/tables/ta_main_results.csv');
fig3 = figure('Color','w','Position',[100 100 700 450]);
bar(categorical(T_main.Model, T_main.Model), T_main.R2, ...
    'FaceColor', [0.30 0.56 0.85]);
hold on;
for k = 1:height(T_main)
    text(k, T_main.R2(k)+0.02, sprintf('%.3f', T_main.R2(k)), ...
        'HorizontalAlignment','center', 'FontSize', 11);
end
ylabel('R^2'); ylim([0 1.0]);
grid on; set(gca, 'FontSize', 11);
exportgraphics(fig3, fullfile(outdir, 'ta_fig3_main.png'), 'Resolution', 300);
close(fig3);

%% ===== 图4:VMD消融 =====
T_abl = readtable('results/tables/ta_ablation.csv');
fig4 = figure('Color','w','Position',[100 100 700 450]);
x = [1 2];  w = 0.35;
r2_ridge = T_abl.R2([1 2]);  r2_bst = T_abl.R2([3 4]);
b1 = bar(x-w/2, r2_ridge, w, 'FaceColor', [0.55 0.55 0.55]); hold on;
b2 = bar(x+w/2, r2_bst, w, 'FaceColor', [0.91 0.64 0.24]);
for k = 1:2
    text(x(k)-w/2, r2_ridge(k)+0.015, sprintf('%.3f', r2_ridge(k)), ...
        'HorizontalAlignment','center', 'FontSize', 10);
    text(x(k)+w/2, r2_bst(k)+0.015, sprintf('%.3f', r2_bst(k)), ...
        'HorizontalAlignment','center', 'FontSize', 10);
end
set(gca, 'XTick', x, 'XTickLabel', {'仅手工特征(3维)', '手工+VMD特征(8维)'});
ylabel('R^2'); ylim([0 1.05]);
legend([b1 b2], {'岭回归', '提升树'}, 'Location', 'southeast');
grid on; set(gca, 'FontSize', 11);
exportgraphics(fig4, fullfile(outdir, 'ta_fig4_ablation.png'), 'Resolution', 300);
close(fig4);

%% ===== 图5:窗口长度 =====
T_win = readtable('results/tables/ta_window.csv');
Wv = [0.25 0.5 1.0];                    % 与 ta_window.csv 行顺序一致
fig5 = figure('Color','w','Position',[100 100 700 450]);
plot(Wv, T_win.R2, '-o', 'LineWidth', 1.8, 'MarkerSize', 9, ...
    'MarkerFaceColor', [0.30 0.56 0.85], 'MarkerEdgeColor', 'k');
hold on;
for k = 1:height(T_win)
    text(Wv(k), T_win.R2(k)+0.02, sprintf('%.3f', T_win.R2(k)), ...
        'HorizontalAlignment','center', 'FontSize', 10);
end
xlabel('早预警窗口长度 W (s)'); ylabel('R^2');
xticks(Wv); ylim([0.7 1.05]);
grid on; set(gca, 'FontSize', 11);
exportgraphics(fig5, fullfile(outdir, 'ta_fig5_window.png'), 'Resolution', 300);
close(fig5);

fprintf('5张图已生成至 %s/,请复制到论文 figures/ 目录\n', outdir);
end
