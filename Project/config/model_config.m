function mc = model_config()
% MODEL_CONFIG  模型训练与评估的配置参数
% 大白话:机器学习模型也有"旋钮"(学多快、学多少轮、用多少数据训练),
%        这些旋钮集中放在这里,调参时只动这个文件。

mc.train_ratio = 0.7;        % 70%数据训练,30%测试
mc.seed        = 42;         % 随机种子:保证每次划分结果可复现

% 梯度提升(XGBoost思想,MATLAB统计学习工具箱实现)
mc.xgb_cycles  = 100;        % 基学习器(小树)的数量
mc.xgb_lr      = 0.1;        % 学习率:每棵树的贡献权重,小步慢走更稳

% LSTM(深度学习工具箱)
mc.lstm_hidden = 50;         % 隐层神经元数
mc.lstm_epochs = 100;        % 训练轮数(小样本下每轮batch极少,需要更多轮)

% 鲁棒性测试:信噪比列表(dB),越小噪声越大
mc.snr_list    = [40, 30, 20, 10];

% 热-电耦合:热泵虚拟惯量(等效附加惯性常数,秒)
mc.H_vhp       = 2;
end
