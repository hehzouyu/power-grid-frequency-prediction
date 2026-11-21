%% TEST_DATASET  测试2:验证批量数据集生成
% 测试目标:
%   1) 数据集维度正确:F 是 N×T,nadir 是 N×1,params 是 N×5
%   2) 数据干净:无 NaN、无 Inf
%   3) Nadir 统计合理:所有场景的 Nadir 都应低于50Hz,且数值分布正常
%   4) 抽查曲线:画出几条不同场景曲线,形态应各不相同
%
% 通过标准:下方断言全部通过 + 曲线簇形态肉眼正常

clear; clc; close all;

% 工程根目录加进路径
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
projRoot = fileparts(fileparts(mfilename('fullpath')));  % 工程根目录(绝对路径,不受运行位置影响)

%% 第1步:加载配置并生成数据集(32条,demo规模)
sys = sys_config();
sim = sim_config();
ps  = param_space();

dataset = gen_dataset(sys, sim, ps);

%% 第2步:保存到 data/raw/(正式流程的数据落盘位置)
if ~exist(fullfile(projRoot,'data/raw'), 'dir'); mkdir(fullfile(projRoot,'data/raw')); end
save(fullfile(projRoot,'data/raw/dataset_raw.mat'), 'dataset', 'sys', 'sim', 'ps');
fprintf('数据集已保存至 data/raw/dataset_raw.mat\n');

%% 第3步:断言检查
[N, T] = size(dataset.F);

% 检查1:维度
assert(N == ps.N_case, '场景数不对:F的行数应等于参数空间总数');
assert(T == numel(dataset.t), '曲线长度不对:F的列数应等于时间点数');
assert(numel(dataset.nadir) == N, 'nadir长度不对');
assert(size(dataset.params, 2) == 5, 'params应为N×5(五维参数)');
fprintf('[通过] 维度检查: F = %d×%d, nadir = %d×1, params = %d×5\n', N, T, N, N);

% 检查2:无 NaN / Inf
assert(~any(isnan(dataset.F(:))), 'F中存在NaN!');
assert(~any(isinf(dataset.F(:))), 'F中存在Inf!');
assert(~any(isnan(dataset.nadir)), 'nadir中存在NaN!');
fprintf('[通过] 数据干净,无NaN/Inf\n');

% 检查3:Nadir 统计
assert(all(dataset.nadir < 50), '存在Nadir>=50Hz的场景,扰动可能未生效');
fprintf('[通过] Nadir统计: min = %.4f, max = %.4f, mean = %.4f Hz\n', ...
    min(dataset.nadir), max(dataset.nadir), mean(dataset.nadir));

%% 第4步:抽查画曲线簇(每类扰动各抽几条)
figure('Name', 'test_dataset');
hold on;
idx_plot = round(linspace(1, N, 8));       % 均匀抽8条
for k = 1:numel(idx_plot)
    i = idx_plot(k);
    plot(dataset.t, dataset.F(i, :), 'LineWidth', 1);
end
xlabel('时间 (s)'); ylabel('频率 (Hz)');
title(sprintf('数据集曲线簇抽查(共%d条,展示8条)', N));
grid on;

fprintf('[通过] 测试2完成,请检查曲线簇形态是否各异\n');
