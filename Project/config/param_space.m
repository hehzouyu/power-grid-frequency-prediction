function ps = param_space(mode)
% PARAM_SPACE  参数空间(按助教建议范围, 2026-09-24版)
%
% 输入:
%   mode : 'demo'(默认,32组,测试用) 或 'full'(1200组,正式实验)
%
% 参数范围(助教建议,教学仿真设定,实际电网参数需另行验证):
%   H  : 惯性常数 2~6 s
%   D  : 阻尼系数 0.5~2 pu
%   R  : 下垂系数 0.04~0.06 pu
%   Tg : 调速器时间常数 0.2~1 s
%   d  : 扰动功率缺额 0.02~0.08 pu

if nargin < 1, mode = 'demo'; end

switch mode
    case 'demo'
        % demo模式:2^5 = 32组
        ps.H_list  = [2, 5];
        ps.D_list  = [0.5, 1.5];
        ps.R_list  = [0.04, 0.06];
        ps.Tg_list = [0.2, 1.0];
        ps.dP_list = [0.02, 0.08];

    case 'full'
        % 正式模式:5×4×3×4×5 = 1200组
        ps.H_list  = [2, 3, 4, 5, 6];                    % 惯量:五档
        ps.D_list  = [0.5, 1.0, 1.5, 2.0];               % 阻尼:四档
        ps.R_list  = [0.04, 0.05, 0.06];                 % 下垂:三档
        ps.Tg_list = [0.2, 0.45, 0.7, 1.0];              % 调速器:四档
        ps.dP_list = [0.02, 0.035, 0.05, 0.065, 0.08];   % 缺额:五档

    otherwise
        error('未知模式:%s(可选 demo / full)', mode);
end

ps.mode  = mode;
ps.N_case = numel(ps.H_list)*numel(ps.D_list)*numel(ps.R_list) ...
          * numel(ps.Tg_list)*numel(ps.dP_list);
fprintf('参数空间[%s模式]: %d 个场景\n', mode, ps.N_case);
end
