function results = eval_models(models, X_te, y_te, F_te, params_te, sys)
% EVAL_MODELS  统一评估所有模型,输出RMSE/MAE/R^2对比表
%
% 输入:
%   models    : train_models 的输出
%   X_te      : 测试集特征(表格数据模型用)
%   y_te      : 测试集真值
%   F_te      : 测试集原始曲线(LSTM用)
%   params_te : 测试集参数(解析法基线用)
%   sys       : 系统参数(取额定频率)
%
% 输出 results: 表格,每行一个模型,列为 rmse / mae / r2

% --- 各模型预测 ---
yhat_lin  = predict(models.lin, X_te);
yhat_xgb  = predict(models.xgb, X_te);
% LSTM评估:必须用训练时相同的归一化参数处理测试曲线,
% 否则输入分布不一致,预测会完全失真
F_te_norm = (F_te - models.norm(1)) / models.norm(2);
yhat_lstm_n = predict(models.lstm, num2cell(F_te_norm, 2));
% 训练目标是归一化的,预测值须反变换回Hz单位
yhat_lstm = yhat_lstm_n * models.normy(2) + models.normy(1);
yhat_ana  = baseline_analytic(params_te, sys);

% --- 逐个计算指标 ---
m_lin  = calc_metrics(y_te, yhat_lin);
m_xgb  = calc_metrics(y_te, yhat_xgb);
m_lstm = calc_metrics(y_te, yhat_lstm);
m_ana  = calc_metrics(y_te, yhat_ana);

% --- 汇总成表 ---
model_names = {'解析法基线'; '线性回归'; 'XGBoost'; 'LSTM'};
rmse_all = [m_ana.rmse; m_lin.rmse; m_xgb.rmse; m_lstm.rmse];
mae_all  = [m_ana.mae;  m_lin.mae;  m_xgb.mae;  m_lstm.mae];
r2_all   = [m_ana.r2;   m_lin.r2;   m_xgb.r2;   m_lstm.r2];

results = table(model_names, rmse_all, mae_all, r2_all, ...
    'VariableNames', {'Model', 'RMSE_Hz', 'MAE_Hz', 'R2'});

% 预测值一并返回,方便后续画图
% 注意必须是 4×1 的列cell(与表的高度4匹配),写成 1×4 会报维度错误
results.yhat = {yhat_ana; yhat_lin; yhat_xgb; yhat_lstm};
end
