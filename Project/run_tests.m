function run_tests()
% RUN_TESTS  测试总入口:顺序运行全部五个模块测试
% 大白话:每个模块都有自己的"验收单"(tests/ 下的测试脚本),
%        这个脚本把所有验收单按依赖顺序连起来跑一遍:
%        仿真 → 数据集 → 特征 → 模型 → 鲁棒性。
%        任何一步报错会在这里直接暴露,方便定位问题。
%
% 重要实现说明:测试脚本内部有 clear 命令,会清空当前工作区。
% 如果本脚本用普通变量保存测试清单,第一次 run() 之后清单就被清掉了。
% 因此清单通过局部函数 getTestFiles() 在【每次循环时重新获取】,
% 局部函数有自己独立的工作区,不受 clear 影响。
%
% 用法: run_tests  或  run_tests()

clear; clc; close all;
addpath(genpath(pwd));          % 整个工程加进路径

for i = 1:numel(getTestFiles())   % 循环上限每次重新计算(防被 clear 清掉)
    files = getTestFiles();     % 每次重新获取清单(防被 clear 清掉)
    n_tests = numel(files);     % 总数也每次重算
    fprintf('\n############################################################\n');
    fprintf('运行测试 %d/%d: %s\n', i, n_tests, files{i});
    fprintf('############################################################\n');
    run(files{i});
end

fprintf('\n全部测试通过!\n');
end

function files = getTestFiles()
% 测试清单(集中在这里,增删测试只改这一个地方)
files = { ...
    'tests/test_simulate.m', ...   % 模块1:单次仿真
    'tests/test_dataset.m', ...    % 模块2:批量数据集
    'tests/test_features.m', ...   % 模块3:VMD特征
    'tests/test_models.m', ...     % 模块4:模型训练
    'tests/test_noise.m'};         % 模块5:鲁棒性
end
