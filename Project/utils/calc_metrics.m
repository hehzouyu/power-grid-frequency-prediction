function m = calc_metrics(y_true, y_pred)
% CALC_METRICS  回归评估指标(助教版:增加绝对误差95百分位数与耗时记录位)
%
% 指标含义:
%   RMSE : 均方根误差(Hz),大误差敏感
%   MAE  : 平均绝对误差(Hz),直观
%   P95  : 绝对误差的95百分位数(Hz)——"最坏情况"指标,助教要求
%   R2   : 决定系数
%   time_s: 训练/推理耗时(秒),由调用方赋值,默认为NaN

y_true = y_true(:); y_pred = y_pred(:);
err = y_pred - y_true;

m.rmse  = sqrt(mean(err.^2));
m.mae   = mean(abs(err));
m.p95   = prctile(abs(err), 95);
m.r2    = 1 - sum(err.^2)/sum((y_true - mean(y_true)).^2);
m.time_s = NaN;   % 耗时由调用方填入
end
