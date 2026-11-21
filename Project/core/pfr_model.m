function p_target = pfr_model(x, sys)
% PFR_MODEL  一次调频目标功率(聚合模型的限幅下垂特性)
%
% 大白话:频率一跌(x<0),调速器按下垂系数1/R要求增加出力,
%        但出力被限制在[0, p_max]之间(备用容量有上限)。
%        与旧版区别:这里只算"目标值",惯性环节(Tg)由摇摆方程积分处理。
%
% 输入:
%   x   : 无量纲频偏,负值表示频率跌落
%   sys : 系统参数(需要 R / p_max)
%
% 输出:
%   p_target : 调频目标功率(pu),范围 [0, p_max]

p_target = min( max(-x / sys.R, 0), sys.p_max );
end
