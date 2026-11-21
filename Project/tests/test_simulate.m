%% TEST_SIMULATE  测试1:验证单次仿真的正确性
% 测试目标:
%   1) 频率曲线呈现正确的三段形态:平稳 → 跌落 → 回升
%   2) Nadir(最低点)数值合理(应明显低于50Hz,但又不至于离谱)
%   3) 扰动发生时刻(t=1s)之前,频率应保持50Hz不变
%
% 通过标准(肉眼检查 + 命令行输出):
%   - 图上能看到 1秒处频率开始下跌,跌到谷底后被拉回
%   - 命令行打印的 Nadir 在合理范围(大约 49.90 ~ 49.99 Hz 之间,
%     PFR正确动作后跌落应被明显抑制)

clear; clc; close all;

% 把工程根目录加进路径,这样 config/core/utils 里的函数都能被找到
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));

%% 第1步:加载配置
sys = sys_config();      % 电网参数
sim = sim_config();      % 仿真设置

%% 第2步:执行仿真
[t, f, nadir] = simulate_freq(sys, sim);

%% 第3步:命令行输出关键结果
fprintf('==== 测试1:单次频率仿真 ====\n');
fprintf('仿真时长: %.1f s, 共 %d 个点\n', sim.t_end, numel(t));
fprintf('扰动: t = %.1f s 时刻,失去 %.2f pu 发电功率\n', sim.t_dist, sim.dP);
fprintf('Nadir(最低点频率) = %.4f Hz\n', nadir);
fprintf('最大跌落深度      = %.4f Hz\n', sys.f_nom - nadir);

% 简单断言:扰动前频率必须等于50Hz
% 检查窗口说明:扰动是阶跃信号,ode45在间断点(t=1s)附近的插值误差较大,
% 且与步长有关(实测可超过1e-3 Hz),在这个位置断言数值精度没有意义。
% 因此只检查"扰动前0.1秒以外"的区段——那里系统严格平衡,
% 频率应保持50Hz不变,数值上应精确到机器精度。
idx_before = t < (sim.t_dist - 0.1);
dev_before = max(abs(f(idx_before) - sys.f_nom));
fprintf('扰动前(t<%.1fs)最大偏差 = %.2e Hz (应接近机器精度)\n', ...
    sim.t_dist - 0.1, dev_before);
assert(dev_before < 1e-6, '扰动前频率不为50Hz,仿真有误!');
fprintf('[通过] 扰动前频率保持50Hz\n');

% 简单断言:Nadir必须低于额定频率(确实发生了跌落)
assert(nadir < sys.f_nom, 'Nadir不低于50Hz,扰动未生效!');
fprintf('[通过] Nadir < 50Hz,跌落正常发生\n');

%% 第4步:画图检查
figure('Name', 'test_simulate');
plot(t, f, 'b-', 'LineWidth', 1.5); hold on;
yline(sys.f_nom, '--k', '额定频率 50Hz');
xline(sim.t_dist, '--r', '扰动时刻');
plot(t(f == nadir), nadir, 'ro', 'MarkerFaceColor', 'r');   % 标出最低点
xlabel('时间 (s)'); ylabel('频率 (Hz)');
title('频率响应曲线:跌落 → Nadir → 回升');
legend('频率曲线', '额定频率', '扰动时刻', 'Nadir');
grid on;

fprintf('[通过] 测试1完成,请检查图形是否符合三段形态\n');
