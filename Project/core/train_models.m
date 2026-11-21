function models = train_models(X_tr, y_tr, F_tr, mc)
% TRAIN_MODELS  训练三个预测模型(线性回归 / 梯度提升 / LSTM)
%
% 大白话:三个模型各有角色——
%   线性回归:最简单的基线,"大概齐"能到什么程度,给它把尺子
%   梯度提升(XGBoost思想):主力选手,擅长捕捉特征之间的复杂配合
%   LSTM:端到端选手,不看我们手工设计的特征,直接"啃"原始曲线,
%         用来检验手工特征到底有没有价值
%
% 输入:
%   X_tr : n_train × d   VMD特征+传统参数(表格数据模型用)
%   y_tr : n_train × 1   Nadir标签
%   F_tr : n_train × T   原始频率曲线(LSTM用)
%   mc   : 模型配置(model_config 的输出)
%
% 注意:models.norm 记录了LSTM输入的归一化参数,eval_models 评估时必须复用
%
% 输出 models: 结构体,字段 lin / xgb / lstm

rng(mc.seed);   % 固定随机种子,结果可复现

% --- 模型1:多元线性回归 ---
% fitlm 是 MATLAB 统计学习工具箱内置函数
% 注意:若出现"回归设计矩阵秩亏"警告,属于预期现象——
%       VMD特征中各IMF能量占比之和恒为1(成分数据天然共线),
%       与截距项组合后出现完全线性相关。这只是警告不影响使用;
%       如需彻底消除,可在特征工程阶段去掉最后一个能量占比列。
models.lin = fitlm(X_tr, y_tr);

% --- 模型2:梯度提升(与XGBoost同类的boosting思想) ---
% fitrensemble + LSBoost 是 MATLAB 内置的回归提升实现:
% 一串小决策树,每棵专门补前一棵的短板,逐步逼近目标
models.xgb = fitrensemble(X_tr, y_tr, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', mc.xgb_cycles, ...
    'LearnRate', mc.xgb_lr);

% --- 模型3:LSTM 时序模型 ---
% 关键:必须先归一化!原始频率约50Hz,量纲太大,LSTM的激活函数和
%       梯度更新对该量级极不友好,网络一开始就会饱和,训练彻底失败
%       (实测不归一化时RMSE高达48Hz,等于完全没学到)。
%       做法:z-score标准化,并把参数存进 models.norm 供评估时复用。
mu_f = mean(F_tr(:));  sg_f = std(F_tr(:));
models.norm = [mu_f, sg_f];                  % 记录输入变换参数
F_norm = (F_tr - mu_f) / sg_f;

% 目标(Nadir)也必须归一化!原因:22条样本=每轮仅1个batch,
% 30轮训练总共只更新30次;若目标保持~50Hz的原始量级,
% 网络输出要从0爬到50,30步根本走不到(实测输出卡在5Hz,RMSE≈45)。
% 目标z-score后起步即接近正确答案,同样的轮数能真正收敛。
mu_y = mean(y_tr);  sg_y = std(y_tr);
models.normy = [mu_y, sg_y];                 % 记录目标变换参数
y_norm = (y_tr - mu_y) / sg_y;

% 数据形态转换:每条曲线看作一个"序列样本",转成 cell 数组
% (LSTM 的输入不是表格,而是一段按时间排列的信号)
X_seq = num2cell(F_norm, 2);        % n×1 cell,每个元素 1×T

layers = [ ...
    sequenceInputLayer(1, 'Name', 'input')                 % 输入:单通道时序
    lstmLayer(mc.lstm_hidden, 'OutputMode', 'last', ...    % LSTM:读完整条曲线,
              'Name', 'lstm')                              %   最后一步输出总结
    fullyConnectedLayer(1, 'Name', 'fc')                   % 映射到一个数:Nadir
    regressionLayer('Name', 'out') ];                      % 回归输出层

opts = trainingOptions('adam', ...                         % Adam优化器
    'MaxEpochs', mc.lstm_epochs, ...
    'GradientThreshold', 1, ...       % 梯度截断:防止小数据下梯度爆炸
    'Verbose', false, ...
    'Plots', 'none');

models.lstm = trainNetwork(X_seq, y_norm, layers, opts);   % 用归一化目标训练
end
