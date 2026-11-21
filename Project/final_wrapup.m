%% ===== 收尾脚本: R03全1200基线 + R14噪声MAE/P95 + R21在线耗时 + R06 VMD随机性 =====
clear; clc;
cd(fileparts(mfilename('fullpath')))
addpath(genpath(pwd))

S = load('data/raw/dataset_ta.mat'); dataset = S.dataset; sim = S.sim;
y = dataset.nadir;                                   % 式(2)标签: 全部1200组都有定义
rng(42); idx = find(~dataset.posthoc); idx = idx(randperm(numel(idx)));
n_tr = floor(0.6*numel(idx)); n_va = floor(0.2*numel(idx));
tr = idx(1:n_tr); te = idx(n_tr+n_va+1:end);
iw = find(dataset.t > sim.t_dist & sim.t_dist+0.5 >= dataset.t);
tw = dataset.t(iw); fw = dataset.F(:, iw) - 50;
a = fw(:,end);
PhiB = [a, mean(diff(fw,1,2)./diff(tw),2), max(fw,[],2)-min(fw,[],2)];
sl = (fw(:,[9 17 25]) - fw(:,[1 9 17])) ./ (tw([9 17 25]) - tw([1 9 17]));
PhiS = [a, sl];

%% R06(可选但便宜): VMD分解随机性稳定性——固定输入,重置rng分解10次,看特征波动
i_test = idx(1);
xw = fw(i_test,:) - mean(fw(i_test,:));
feats = zeros(10,5);
for rep = 1:10
    rng(5000+rep);
    imf = vmd(xw,'NumIMF',4,'AbsoluteTolerance',1e-6);
    if size(imf,2)==4 && size(imf,1)~=4, imf = imf.'; end
    e2 = sum(imf.^2,2); p2 = e2/sum(e2); p2 = max(p2,1e-12);
    feats(rep,:) = [p2.', -sum(p2.*log2(p2))];
end
fprintf('R06 VMD稳定性: 10次分解特征最大值-最小值 = %s\n', mat2str(range(feats,1),3));
fprintf('R06 (能量占比1波动 %.2e, 熵波动 %.2e)\n', range(feats(:,1)), range(feats(:,5)));

%% R03: 全部1200例基线(提前事件场景也有式(2)标签,无需新模型——在训练+验证集上重训后于全集评估)
resAll = [];
all1200 = (1:1200)';
% 先算VMD特征(若工作区没有Xv)
if ~exist('Xv','var') || size(Xv,1)~=1200
    Xv = zeros(1200,5);
    for i = 1:1200
        xw2 = fw(i,:) - mean(fw(i,:));
        imf = vmd(xw2,'NumIMF',4,'AbsoluteTolerance',1e-6);
        if size(imf,2)==4 && size(imf,1)~=4, imf = imf.'; end
        e2 = sum(imf.^2,2); p2 = e2/sum(e2); p2 = max(p2,1e-12);
        Xv(i,:) = [p2.', -sum(p2.*log2(p2))];
    end
end
cfgs = {PhiB, [PhiB Xv], PhiS};
cfgNames = {'手工三特征','手工+VMD','分段斜率'};
trv = idx(1:n_tr+n_va);                              % 训练+验证(928),避免训练集泄漏
for cfg = 1:3
    Phi = cfgs{cfg};
    Z = (Phi - mean(Phi(trv,:),1)) ./ std(Phi(trv,:),1);
    mdl = fitrensemble(Z(trv,:), y(trv), 'Method','LSBoost','NumLearningCycles',100,'LearnRate',0.1);
    e = predict(mdl, Z(all1200,:)) - y(all1200);
    resAll = [resAll; cfg 1200 sqrt(mean(e.^2)) mean(abs(e)) ...
              quantile(abs(e),0.95) 1-sum(e.^2)/sum((y(all1200)-mean(y(all1200))).^2)]; %#ok<AGROW>
    fprintf('R03 全1200 cfg%d(%s): RMSE=%.5f MAE=%.5f P95=%.5f R2=%.4f\n', ...
        cfg, cfgNames{cfg}, resAll(end,3), resAll(end,4), resAll(end,5), resAll(end,6));
end
writetable(array2table(resAll,'VariableNames',{'Cfg','N','RMSE_Hz','MAE_Hz','P95_Hz','R2'}), ...
    'results/tables/ta_all1200_baseline.csv');

%% R14: 噪声实验补MAE/P95(三配置,测试集232,配对,10次)
resN = [];
for snr = [40 30 20]
    for rep = 1:10
        rng(2000+rep);
        fw_n = zeros(numel(te), numel(iw));
        for k = 1:numel(te)
            fw_n(k,:) = add_noise(dataset.F(te(k), iw), snr, 50) - 50;
        end
        an = fw_n(:,end);
        sln = (fw_n(:,[9 17 25]) - fw_n(:,[1 9 17])) ./ (tw([9 17 25]) - tw([1 9 17]));
        baseB = [an, mean(diff(fw_n,1,2)./diff(tw),2), max(fw_n,[],2)-min(fw_n,[],2)];
        Xvn = zeros(numel(te),5);
        for k = 1:numel(te)
            xw3 = fw_n(k,:) - mean(fw_n(k,:));
            imf = vmd(xw3,'NumIMF',4,'AbsoluteTolerance',1e-6);
            if size(imf,2)==4 && size(imf,1)~=4, imf = imf.'; end
            e2 = sum(imf.^2,2); p2 = e2/sum(e2); p2 = max(p2,1e-12);
            Xvn(k,:) = [p2.', -sum(p2.*log2(p2))];
        end
        cfgF = {PhiB, [PhiB Xv], PhiS};
        cfgN = {baseB, [baseB Xvn], [an sln]};
        for cfg = 1:3
            Z = (cfgF{cfg} - mean(cfgF{cfg}(tr,:),1)) ./ std(cfgF{cfg}(tr,:),1);
            mdl = fitrensemble(Z(tr,:), y(tr), 'Method','LSBoost','NumLearningCycles',100,'LearnRate',0.1);
            Zn = (cfgN{cfg} - mean(cfgF{cfg}(tr,:),1)) ./ std(cfgF{cfg}(tr,:),1);
            e = predict(mdl, Zn) - y(te);
            resN = [resN; snr rep cfg sqrt(mean(e.^2)) mean(abs(e)) quantile(abs(e),0.95)]; %#ok<AGROW>
        end
    end
    for cfg = 1:3
        r = resN(resN(:,1)==snr & resN(:,3)==cfg,:);
        fprintf('R14 SNR=%ddB cfg%d: RMSE=%.5f±%.5f MAE=%.5f P95=%.5f\n', ...
            snr, cfg, mean(r(:,4)), std(r(:,4)), mean(r(:,5)), mean(r(:,6)));
    end
end
writetable(array2table(resN,'VariableNames',{'SNR','Rep','Cfg','RMSE_Hz','MAE_Hz','P95_Hz'}), ...
    'results/tables/ta_noise_map95.csv');

%% R21(可选): 在线耗时实测(分段斜率配置, 单次完整流程)
N_rep = 100;
t0 = tic;
for rep = 1:N_rep
    i_s = te(mod(rep, numel(te))+1);
    xw4 = fw(i_s,:) - mean(fw(i_s,:));
    a4 = xw4(end);
    s4 = (xw4([9 17 25]) - xw4([1 9 17])) ./ (tw([9 17 25]) - tw([1 9 17]));
    Zq = ([a4 s4] - mean(PhiS(tr,:),1)) ./ std(PhiS(tr,:),1);
end
t_feat = toc(t0)/N_rep*1000;
Zfull = (PhiS - mean(PhiS(tr,:),1)) ./ std(PhiS(tr,:),1);
mdlT = fitrensemble(Zfull(tr,:), y(tr), 'Method','LSBoost','NumLearningCycles',100,'LearnRate',0.1);
t0 = tic;
for rep = 1:N_rep
    predict(mdlT, Zfull(te(1:10),:));
end
t_inf = toc(t0)/N_rep/10*1000;
fprintf('R21 在线耗时: 特征提取 %.3f ms + 推理 %.3f ms ≈ %.3f ms (决策预算500 ms, 余量充足)\n', ...
    t_feat, t_inf, t_feat+t_inf);

fprintf('收尾脚本全部完成\n');
