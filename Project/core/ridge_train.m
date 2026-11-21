function [mdl, lambda_best] = ridge_train(Z_tr, y_tr, Z_val, y_val, lambda_grid)
% RIDGE_TRAIN  岭回归(助教规范:L2正则,截距不惩罚,λ由独立验证集选择)
%
% 求解: min 1/n·||y - θ0 - Zθ||² + λ||θ||²   (θ0 不惩罚)
% 采用解析解(特征维度小,无需迭代):
%   θ = (A'A + P)^{-1} A'y, A=[1,Z], P=diag(0,λ,...,λ)
%
% 输入:
%   Z_tr, y_tr   : 训练集(特征已用训练集统计量标准化)
%   Z_val, y_val : 验证集(λ选择用)
%   lambda_grid  : 候选λ(默认 1e-4 ~ 1e2,对数网格)
%
% 输出:
%   mdl        : 结构体(theta/lambda)
%   lambda_best: 验证集RMSE最小的λ

if nargin < 5
    lambda_grid = 10.^(-4:0.5:2);
end

% 第1步:训练集上逐λ拟合,验证集选优
best_rmse = inf;  lambda_best = lambda_grid(1);
for lam = lambda_grid
    th = ridge_fit(Z_tr, y_tr, lam);
    A_val = [ones(size(Z_val,1),1), Z_val];
    rmse_v = sqrt(mean((y_val - A_val*th).^2));
    if rmse_v < best_rmse
        best_rmse = rmse_v;  lambda_best = lam;
    end
end

% 第2步:用训练+验证集重拟合(λ已固定)
mdl.theta  = ridge_fit([Z_tr; Z_val], [y_tr; y_val], lambda_best);
mdl.lambda = lambda_best;
end

function th = ridge_fit(Z, y, lam)
n = size(Z, 1);
A = [ones(n,1), Z];
P = diag([0; lam * ones(size(Z,2),1)]);
th = (A'*A + P) \ (A'*y);
end
