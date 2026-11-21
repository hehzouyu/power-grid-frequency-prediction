function f_noisy = add_noise(f, snr_db, f_nom)
% ADD_NOISE  在频率曲线上叠加高斯白噪声,模拟量测不确定性
%
% 大白话:真实电网里,PMU量测、通信环节都会给频率曲线带来噪声。
%        信噪比SNR描述"信号多干净":40dB很干净,10dB相当脏。
%        我们在干净的仿真曲线上人为加不同强度的噪声,
%        测试模型在"脏数据"上还能不能保持精度——这叫鲁棒性测试。
%
% 输入:
%   f      : 原始频率曲线(Hz)
%   snr_db : 目标信噪比(dB),越大噪声越小
%   f_nom  : 额定频率(Hz),噪声功率相对"频率偏差"计算
%
% 输出:
%   f_noisy : 加噪后的频率曲线(Hz)

if nargin < 3, f_nom = 50; end
f = f(:);
sig_pow = mean((f - f_nom).^2);                 % 信号功率(以偏差计)
noi_pow = sig_pow / 10^(snr_db/10);             % 由SNR反推噪声功率
f_noisy = f + sqrt(noi_pow) * randn(size(f));   % 叠加高斯白噪声
end
