% 只读复核本轮沿用的固定截止结果，所有误差由成员预测重新计算。
% 运行失败时 assert 将保留具体字段错误；仅在全部核对后写出交付CSV。
root = fileparts(mfilename('fullpath'));
base = fullfile(root,"evidence","nadir","quality-evidence");
get = @(f) readtable(f,"TextType","string");
q = get(fullfile(base,"ecgr","pair-predictions.csv"));
s = get(fullfile(base,"quality","predictions.csv"));
p = get(fullfile(base,"quality","pair-predictions.csv"));
old = get(fullfile(root,"evidence","nadir","pair-evidence","pair-predictions.csv"));
records = cell(0,6);
for model = ["ecgr","quadratic"]
    for condition = ["clean","white30"]
        allRows = q(q.Feature==model & q.Condition==condition,:);
        eligible = allRows(allRows.Status=="primary_eligible",:);
        % 候选减参考；误差先以Hz相减，最后一次性转成mHz。
        difference = allRows.PredictionB_Hz-allRows.PredictionA_Hz;
        assert(max(abs(difference-allRows.DeltaPrediction_Hz))<1e-10);
        direction = sign(difference).*(abs(difference)>0.0005);
        actual = sign(allRows.DeltaTarget_Hz).*(abs(allRows.DeltaTarget_Hz)>0.0005);
        assert(all(direction==allRows.PredictedDirection & actual==allRows.TrueDirection));
        e = 1000*abs(eligible.PredictionB_Hz-eligible.PredictionA_Hz-eligible.DeltaTarget_Hz);
        records(end+1,:) = {"ECGR_"+condition,model,mean(e),percent95(e),sum(direction==actual),height(allRows)};
        assert(numel(unique(allRows.ReferenceID))==96 && height(eligible)==222*(1+9*(condition=="white30")));
    end
end
assert(numel(unique(p.ReferenceID))==96 && numel(unique(p.PairID))==288);
assert(nnz(p.Condition=="clean" & p.Status=="primary_eligible")==220);
s = s(s.Kind=="single",:);
assert(numel(unique(s.Entity))==128);
p = p(p.Status=="primary_eligible",:);
for model = ["gate","endpoint"]
    for group = ["single","pair"]
        if group=="single"
            t = s; label = t.Target_Hz;
        else
            t = p; label = t.DeltaTarget_Hz;
        end
        condition = unique(t.Condition);
        err = 1000*abs(t.(model)-label);
        rate = arrayfun(@(k) mean(err(t.Condition==condition(k))),1:numel(condition));
        assert(numel(condition)==6);
        records(end+1,:) = {"Gate_"+group,model,mean(rate),NaN,NaN,height(t)};
        if group=="single"
            assert(round(rate(condition=="jump"),2)==24.86+0.09*(model=="gate"));
        end
    end
end
z = cell2table(records,"VariableNames",["Study","Strategy","MAE_mHz","P95_mHz","Correct","EvaluatedRows"]);
assert(all(abs(z.MAE_mHz-[4.4448513153;13.9583242964;5.6715515014;13.7059;19.2498296233;13.4347976005;23.9262945965;15.5741831596]) < [1e-7;1e-7;1e-7;0.01;1e-7;1e-7;1e-7;1e-7]));
assert(z.Correct(1)==234 && z.Correct(3)==246);
assert(z.P95_mHz(2)>36.043381186205984);
old = old(old.Feature=="endpoint" & old.Condition=="clean",:);
assert(height(old)==192 && nnz(old.Status=="primary_eligible")==148 && nnz(old.TrueDirection==0)==1);
folders = [fullfile(base,"quality"),fullfile(base,"ecgr"),fullfile(root,"evidence","rolling-ecgr"),fullfile(root,"evidence","risk")];
status = zeros(4,2);
for k=1:4
    rules = get(fullfile(folders(k),"acceptance.csv"));
    status(k,:) = [nnz(lower(string(rules.Pass))=="true"),height(rules)];
end
assert(isequal(status,[20 20;7 8;3 4;10 21]));
writetable(z,fullfile(root,"复算表","固定截止证据复算.csv"));
fprintf("PASS: ECGR/quality gate predictions, original pairs and archived acceptance counts.\n");

function q=percent95(x)
    % NumPy 默认线性分位数；明确插值避免 MATLAB 版本间默认定义差异。
    y=sort(x); index=1+0.95*(numel(y)-1);
    a=floor(index); b=ceil(index);
    q=y(a)+(index-a)*(y(b)-y(a));
end

