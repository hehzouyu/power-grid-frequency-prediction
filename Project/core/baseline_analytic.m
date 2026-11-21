function nadir_est = baseline_analytic(params, sys)
% BASELINE_ANALYTIC  解析法基线(新模型:准稳态近似)
% 稳态: p = -x/R (未限幅时), 0 = -d - D*x + p
%   => x_ss = -d / (1/R + D)
% 限幅: 若 -x_ss/R > p_max 则饱和,x_ss = -(d - p_max)/D (若 d > p_max)
%
% 输入 params: N×5 [H, D, R, Tg, dP](H/Tg 本基线不用)
% 输出: N×1 解析估计的最低点近似(Hz,取准稳态值)

D = params(:,2);  R = params(:,3);  d = params(:,5);
p_max = sys.p_max;

x_ss = -d ./ (1./R + D);
% 调频饱和修正:要求出力超过备用上限时,缺口只能由阻尼承担
sat = (-x_ss ./ R) > p_max;
x_ss(sat) = -(d(sat) - p_max) ./ D(sat);

nadir_est = sys.f_nom * (1 + x_ss);
end
