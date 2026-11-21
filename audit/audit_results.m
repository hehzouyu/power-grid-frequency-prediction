function audit_results
% 审计已冻结滚动证据。输入为只读档案的CSV副本和可复现的事件簇抽样。
% 先独立按保存阈值重放触发，再输出绝对指标、成对区间、诊断和原滚动分解。
% 键、分母、参考结果或同批参数不一致即报错；阈值附近的浮点边界单列。
here=fileparts(mfilename('fullpath')); in=fullfile(here,'复算输入');
out=fullfile(here,'复算表'); if ~isfolder(out), mkdir(out); end
rd=@(x) readtable(fullfile(in,x),'TextType','string');
S=rd('test-scores.csv'); W=rd('test-windows.csv'); C=rd('cutoffs.csv');
A=rd('calibration-windows.csv');
for name=["W","A"]
    if name=="W", W.LowPreQuality=boolfield(W.LowPreQuality); else, A.LowPreQuality=boolfield(A.LowPreQuality); end
end
 NS=rd('test-scenarios.csv'); NC=rd('calibration-scenarios.csv');
old=rd('original-events.csv'); OW=rd('original-windows.csv');
known=rd('paired-effects.csv'); knownAbs=rd('event-metrics.csv');
design=jsondecode(fileread(fullfile(here,'事后诊断规则.json')));
p=jsondecode(fileread(fullfile(in,'protocol.json')));
times=[.2 .3 .4 .5];
conditions=string(p.conditions); main=string(p.calibration_conditions);
scopes=["mixed",conditions(:).']; methods=unique(S.Policy,'stable');
n=height(NS); weights=load(fullfile(in,'frozen_draws.mat'),'draws'); draws=double(weights.draws);
assert(n==2400 && height(NC)==800 && height(S)==288000);
assert(height(unique(S(:,{'Event','Condition','Policy','TargetFAR'}),'rows'))==height(S));
assert(height(unique(W(:,{'Event','Condition','Window_s'}),'rows'))==height(W));
assert(height(unique(A(:,{'Event','Condition','Window_s'}),'rows'))==height(A));
% x按事件和窗排序；分级模式沿用同一个扰动前质量标记。
replay=zeros(0,5); calAudit=zeros(0,5); epsHz=3e-13;
W=sortrows(W,{'Condition','Event','Window_s'});
A=sortrows(A,{'Condition','Event','Window_s'});
for m=1:numel(methods)
    policy=methods(m); [field,js,strat]=source(policy);
    for t=p.far_targets(:).'
        cuts=C(C.Policy==policy & abs(C.TargetFAR-t)<1e-10,:);
        for row=1:height(cuts)
            group=cuts.Group(row);
            b=A(ismember(A.Condition,main)&isinf(A.Crossing_s),:);
            if group~="all", b=b(b.LowPreQuality==(group=="low_quality"),:); end
            if numel(js)==1, b=b(abs(b.Window_s-.5)<1e-12,:); end
            ids=findgroups(b.Condition,b.Event);
            ranks=sort(splitapply(@min,b.(field),ids));
            assert(numel(ranks)==cuts.NegativeVariants(row));
            rank=1+floor(t*numel(ranks));
            cut=cuts.Cutoff_Hz(row);
            assert(abs(ranks(rank)-cut)<epsHz);
            lo=sum(ranks<cut-epsHz); hi=sum(ranks<cut+epsHz);
            assert(lo<=cuts.Triggered(row) && cuts.Triggered(row)<=hi);
            calAudit(end+1,:)=[m,t,row,lo,hi]; %#ok<AGROW>
        end
        for cond=conditions(:).'
            b=W(W.Condition==cond,:); q=sortrows(S(S.Policy==policy & S.Condition==cond & abs(S.TargetFAR-t)<1e-10,:),'Event');
            pred=reshape(b.(field),4,[]).'; truth=reshape(b.Crossing_s,4,[]).';
            pre=reshape(logical(b.LowPreQuality),4,[]).';
            assert(all(pre==pre(:,1),'all') && isequaln(truth,repmat(q.Crossing_s,1,4)));
            assert(isequal(q.Event,(0:n-1).') && isequaln(q.Crossing_s,NS.Crossing_s));
            if strat
                want=repmat(cuts.Cutoff_Hz(cuts.Group=="high_quality"),n,1);
                want(pre(:,1))=cuts.Cutoff_Hz(cuts.Group=="low_quality");
            else
                want=repmat(cuts.Cutoff_Hz(cuts.Group=="all"),n,1);
            end
            assert(max(abs(q.Cutoff_Hz-want))<epsHz);
            emit=repmat(times(js),n,1); fire=pred(:,js)<q.Cutoff_Hz;
            e=emit; e(~fire)=Inf; a=min(e,[],2);
            mismatch=a~=q.Alarm_s;
            if any(mismatch)
                % IEEE浮点的最后几位可能改变严格不等式。只接受能由阈值
                % 附近的预测解释的边界，并保留冻结账本原始触发时刻。
                assert(all(any(abs(pred(mismatch,js)-q.Cutoff_Hz(mismatch))<epsHz,2)));
                for k=find(mismatch).'
                    replay(end+1,:)=[m,t,find(conditions==cond),q.Event(k),a(k)]; %#ok<AGROW>
                end
            end
            risk=isfinite(q.Crossing_s); alarm=isfinite(q.Alarm_s);
            hit=risk&q.Alarm_s<q.Crossing_s; false=~risk & alarm;
            late=risk&alarm&~hit; none=risk&~alarm;
            assert(isequal(hit,q.Outcome=="hit") && isequal(false,q.Outcome=="false_alarm"));
            assert(isequal(late,q.Outcome=="late") && isequal(none,q.Outcome=="no_alarm"));
            assert(max(abs(q.Lead_ms(hit)-1000*(q.Crossing_s(hit)-q.Alarm_s(hit))),[],'all')<1e-8);
        end
    end
end
writetable(array2table(replay,'VariableNames',{'PolicyIndex','TargetFAR','ConditionIndex','Event','AuditAlarm_s'}),fullfile(out,'触发浮点边界.csv'));
writetable(array2table(calAudit,'VariableNames',{'PolicyIndex','TargetFAR','GroupIndex','StrictMin','StrictMax'}),fullfile(out,'校准阈值核对.csv'));
rows=cell(0,22); eff=cell(0,12);
cmp=["gate","quadratic";"gate","persistence";"gate","endpoint";"gate","ecgr";"gate","gate_final_only";"gate_stratified","quadratic_stratified"];
for scope=scopes
    if scope=="mixed", use=main; else, use=scope; end
    for t=p.far_targets(:).'
        s=S(abs(S.TargetFAR-t)<1e-10 & ismember(S.Condition,use),:);
        V=cell(numel(methods),1);
        for m=1:numel(methods)
            policy=methods(m); b=s(s.Policy==policy,:); [v,lead]=event_counts(b,n);
            V{m}=v;
            hit=sum(v(:,2)); risk=sum(v(:,1)); fa=sum(v(:,4)); non=sum(v(:,3));
            co=C(C.Policy==policy & abs(C.TargetFAR-t)<1e-10,:);
            cuts=[co.Cutoff_Hz(co.Group=="all"), NaN,NaN];
            if height(co)>1
                cuts(2)=co.Cutoff_Hz(co.Group=="high_quality"); cuts(3)=co.Cutoff_Hz(co.Group=="low_quality");
            end
            a=knownAbs(knownAbs.Scope==scope & knownAbs.Policy==policy & abs(knownAbs.TargetFAR-t)<1e-10,:);
            assert(isequal([hit,risk,fa,non],[a.Hits,a.RiskVariants,a.FalseAlarms,a.NonriskVariants]));
            assert(max(abs([hit/risk,fa/non,median(lead)]-[a.HitRate,a.FAR,a.LeadMedian_ms]))<1e-9);
            role="事后补表";
            if policy=="gate" || policy=="quadratic"
                if t==.1, role="预设主要比较"; else, role="锁定次要工作点"; end
            elseif endsWith(policy,"stratified"), role="锁定代码内分级诊断"; end
            accept="不适用"; if policy=="gate" && t==.1, accept="主验收未通过"; end
            rows(end+1,:)={scope,t,policy,role,n,numel(use),risk+non,sum(v(:,1)>0),hit,risk,fa,non, ...
                100*hit/risk,100*fa/non,median(lead),q7(lead,.05),cuts(1),cuts(2),cuts(3),accept, ...
                sum(b.Outcome=="late"),sum(b.Outcome=="no_alarm")}; %#ok<AGROW>
        end
        for ix=1:size(cmp,1)
            a=V{methods==cmp(ix,1)}; b=V{methods==cmp(ix,2)};
            br=sum(a(:,1)); bn=sum(a(:,3));
            d=[a(:,1),a(:,2)-b(:,2),a(:,3),a(:,4)-b(:,4)];
            obs=100*[sum(d(:,2))/br,sum(d(:,4))/bn];
            for c=1:4, bs(:,c)=sum(reshape(d(draws,c),size(draws)),2); end %#ok<AGROW>
            hi=100*q7(bs(:,2)./bs(:,1),[.025 .975]); fi=100*q7(bs(:,4)./bs(:,3),[.025 .975]);
            k=known(known.Scope==scope & abs(known.TargetFAR-t)<1e-10 & known.Policy==cmp(ix,1) & known.Comparator==cmp(ix,2),:);
            if height(k)==1, assert(max(abs([obs(1),hi,obs(2),fi]-[k.DeltaHit_pp,k.HitLow95_pp,k.HitHigh95_pp,k.DeltaFAR_pp,k.FARLow95_pp,k.FARHigh95_pp]))<1e-8); end
            eff(end+1,:)={scope,t,cmp(ix,1),cmp(ix,2),obs(1),hi(1),hi(2),obs(2),fi(1),fi(2),br,bn}; %#ok<AGROW>
        end
    end
end
absTable=cell2table(rows,'VariableNames',{'Scope','TargetFAR','Policy','Role','PhysicalEvents','Conditions','Variants','RiskEvents','Hits','RiskVariants','FalseAlarms','NonriskVariants','HitRatePct','FARPct','LeadMedian_ms','LeadP5_ms','PooledCut_Hz','HighQualityCut_Hz','LowQualityCut_Hz','Acceptance','Late','NoAlarmRisk'});
effect=cell2table(eff,'VariableNames',{'Scope','TargetFAR','Policy','Comparator','HitDelta_pp','HitLow_pp','HitHigh_pp','FARDelta_pp','FARLow_pp','FARHigh_pp','RiskVariants','NonriskVariants'});
writetable(absTable,fullfile(out,'全部绝对结果.csv'));
writetable(effect,fullfile(out,'全部配对区间.csv'));
joined=outerjoin(absTable,effect,'Keys',{'Scope','TargetFAR','Policy'},'MergeKeys',true,'Type','left',...
  'RightVariables',{'Comparator','HitDelta_pp','HitLow_pp','HitHigh_pp','FARDelta_pp','FARLow_pp','FARHigh_pp'});
writetable(joined,fullfile(out,'绝对结果与配对区间联合表.csv'));
z=absTable(absTable.Scope=="mixed" & absTable.TargetFAR==.1 & absTable.Policy=="gate",:);
assert(z.PhysicalEvents==2400 && z.Variants==12000 && z.RiskVariants==2045 && z.NonriskVariants==9955 && z.Hits==1622 && z.FalseAlarms==1076);
for pol=["gate","quadratic"]
    pool=absTable(absTable.Scope=="mixed"&absTable.Policy==pol,:);
    for t=p.far_targets(:).'
        a=pool(pool.TargetFAR==t,:); b=absTable(ismember(absTable.Scope,main)&absTable.Policy==pol&absTable.TargetFAR==t,:);
        for key=["Hits","FalseAlarms","RiskVariants","NonriskVariants"], assert(a.(key)==sum(b.(key))); end
        assert(abs(a.HitRatePct-mean(b.HitRatePct))<1e-11 && abs(a.FARPct-mean(b.FARPct))<1e-11);
    end
end
% 使用共用事件ID检查两段风险交集和单策略虚警集合。
split_original(old,out);
swap_clean(W,C,out);
choice=findgroups(W.Condition,W.Window_s,W.Selected);
result=table(splitapply(@(x)x(1),W.Condition,choice),splitapply(@(x)x(1),W.Window_s,choice),...
             splitapply(@(x)x(1),W.Selected,choice),splitapply(@numel,W.Event,choice),...
             'VariableNames',{'Condition','Window_s','Selected','Count'});
result.Fraction=result.Count/n;
writetable(result,fullfile(out,'逐窗选择比例.csv'));
diagnostics(W,OW,old,design,out,main,n);
labels_19_20(A,W,NC,NS,p,out);
fprintf('PASS 已冻结指标、配对区间、逐事件首触发、配对分母与标签边界独立核对。\n');
end

function [field,js,strat]=source(policy)
% 明确区分统计策略名、记录列名与可触发时点。
field=policy; js=1:4; strat=false;
if policy=="quadratic_fixed", field="quadratic"; js=4; end
if endsWith(policy,"stratified"), field=extractBefore(policy,'_stratified'); strat=true; end
end

function [v,lead]=event_counts(b,n)
risk=isfinite(b.Crossing_s); hit=b.Alarm_s<b.Crossing_s & risk; far=isfinite(b.Alarm_s)&~risk;
lead=b.Lead_ms(hit);
v=zeros(n,4);
for k=1:4, x={risk,hit,~risk,far}; v(:,k)=accumarray(b.Event+1,double(x{k}),[n,1]); end
end

function q=q7(v,p)
% 和NumPy线性分位数同定义；忽略非有限值只在诊断无法插值时发生。
x=sort(v(isfinite(v))); if isempty(x),q=nan(size(p));return;end
t=1+(numel(x)-1)*p; l=floor(t); u=ceil(t);
q=reshape(x(l),size(p))+(t-l).*(reshape(x(u),size(p))-reshape(x(l),size(p)));
end

function split_original(old,out)
a=sortrows(old(old.Policy=="quadratic_rolling",:),'Scenario');
b=sortrows(old(old.Policy=="quadratic_fixed",:),'Scenario');
assert(isequal(a.Scenario,b.Scenario) && isequal(a.Crossing_s,b.Crossing_s));
risk=isfinite(a.Crossing_s); early=risk&a.Crossing_s<=.5; late=risk&~early;
h=a.Outcome=="hit"; f=b.Outcome=="hit";
fa=a.Outcome=="false_alarm"; fb=b.Outcome=="false_alarm";
assert(isequal([sum(risk),sum(early),sum(late),sum(h),sum(f),sum(fa),sum(fb)],[291 98 193 272 189 84 74]));
groups=["all_risk","early","late","both_hits","rolling_only","fixed_only","neither_hit","all_nonrisk","both_false","rolling_only_false","fixed_only_false","neither_false"];
mask={risk,early,late,h&f,h&~f,~h&f,risk&~h&~f,~risk,fa&fb,fa&~fb,~fa&fb,~risk&~fa&~fb};
x=cell(numel(groups),10);
for i=1:numel(groups)
    m=mask{i}; la=a.Lead_ms(m&h); lb=b.Lead_ms(m&f);
    x(i,:)={groups(i),sum(m),sum(h&m),sum(f&m),median(la),median(lb),q7(la,.05),q7(lb,.05),sum(fa&m),sum(fb&m)};
end
T=cell2table(x,'VariableNames',{'Group','Events','RollingHits','FixedHits','RollingLeadMedian_ms','FixedLeadMedian_ms','RollingLeadP5_ms','FixedLeadP5_ms','RollingFalse','FixedFalse'});
writetable(T,fullfile(out,'原滚动命中集合.csv'));
end

function swap_clean(W,C,out)
b=W(W.Condition=="clean",:); n=height(b)/4;
truth=reshape(b.Crossing_s,4,[]).'; t=[.2 .3 .4 .5]; risk=isfinite(truth(:,1));
res=zeros(2); far=res; R=cell(0,7); names=["quadratic","gate"];
for i=1:2
    a=reshape(b.(names(i)),4,[]).';
    for j=1:2
        cut=C.Cutoff_Hz(C.Policy==names(j) & C.TargetFAR==.1 & C.Group=="all");
        h=any(a<cut & t<truth,2)&risk; f=any(a<cut,2)&~risk;
        res(i,j)=100*sum(h)/sum(risk); far(i,j)=100*sum(f)/sum(~risk);
        R(end+1,:)={names(i),names(j),cut,sum(h),sum(f),res(i,j),far(i,j)}; %#ok<AGROW>
    end
end
writetable(cell2table(R,'VariableNames',{'Score','CutoffOrigin','Cutoff_Hz','Hits','FalseAlarms','HitPct','FARPct'}),fullfile(out,'干净条件阈值交换.csv'));
step=.5*((res(1,2)-res(1,1))+(res(2,2)-res(2,1)));
representation=.5*((res(2,1)-res(1,1))+(res(2,2)-res(1,2)));
assert(abs(step+representation-(res(2,2)-res(1,1)))<1e-10 && n==2400);
writetable(table(step,representation,res(2,2)-res(1,1),'VariableNames',{'CutoffContribution_pp','ScoreContribution_pp','Total_pp'}),fullfile(out,'干净条件对称分解.csv'));
end

function diagnostics(W,OW,old,design,out,main,n)
% 只在既定5%、10%实际虚警坐标比较；对每个事件簇样本重新找插值括点。
% 及时分数来自真正首次越限以前的窗，零时效事件仍留在分母。
rows=cell(0,13);
rng(573180,'twister'); num=design.bootstrap;
assert(contains(string(num),'4000'));
for dataset=["transfer_mixed","transfer_clean","original_clean"]
    if dataset=="original_clean"
        n0=height(OW)/4;
        OW=sortrows(OW,{'Scenario','Emitted_s'});
        obs=sortrows(old(old.Policy=="quadratic_rolling",:),'Scenario');
        assert(n0==height(obs) && isequal(OW.Scenario,repelem(obs.Scenario,4)));
        t=reshape(OW.Emitted_s,4,[]).';
        a=reshape(OW.quadratic_Hz_at_50,4,[]).';
        b=a; b(:,1:3)=Inf;
        risk=repelem(isfinite(obs.Crossing_s),1);
        truth=obs.Crossing_s; clusters=(1:n0).';
        methods=["原滚动二次拟合","原0.50 s固定二次拟合"];
    else
        n0=n; keep=main; if dataset=="transfer_clean", keep="clean"; end
        args=W(ismember(W.Condition,keep),:);
        t=reshape(args.Window_s,4,[]).';
        a=reshape(args.gate,4,[]).'; b=reshape(args.quadratic,4,[]).';
        truth=args.Crossing_s(1:4:end); risk=isfinite(truth);
        clusters=args.Event(1:4:end)+1;
        methods=["门控滚动","固定二次表示滚动"];
    end
    [OA,FA]=curve_data(a,t,truth,risk,clusters,n0);
    [OB,FB]=curve_data(b,t,truth,risk,clusters,n0);
    R=sparse(ones(sum(risk),1),clusters(risk),1,1,n0);
    U=sparse(ones(sum(~risk),1),clusters(~risk),1,1,n0);
    [pA,vA]=curve_eval(OA,FA,R,U,ones(n0,1),design.x);
    [pB,vB]=curve_eval(OB,FB,R,U,ones(n0,1),design.x);
    d=zeros(4000,numel(design.x));
    for i=1:64:4000
        m=min(64,4001-i);
        sampled=randi(n0,n0,m);
        w=full(sparse(sampled(:),repelem((1:m).',n0),1,n0,m));
        d(i:i+m-1,:)=curve_eval(OA,FA,R,U,w,design.x)-curve_eval(OB,FB,R,U,w,design.x);
    end
    for j=1:numel(design.x)
        x=d(:,j); c=100*q7(x,[.025 .975]);
        rows(end+1,:)={dataset,methods(1),methods(2),100*design.x(j),100*pA(j),100*pB(j),...
            100*(pA(j)-pB(j)),c(1),c(2),sum(isfinite(x)),n0,vA(j),vB(j)}; %#ok<AGROW>
    end
end
writetable(cell2table(rows,'VariableNames',{'Dataset','Policy','Comparator','ActualFAR_pct','PolicyHit_pct','ComparatorHit_pct','Delta_pp','Low_pp','High_pp','ValidDraws','PhysicalEvents','BracketA','BracketB'}),fullfile(out,'同实际虚警事后诊断.csv'));
end

function [O,F]=curve_data(scores,t,truth,risk,cluster,n)
s=scores;
s(t>=truth & risk)=Inf;
m=min(s,[],2); eligible=isfinite(m);
% 等分值成组，防止人为打破并列；非风险事件使用全部决策窗。
[~,~,g]=unique(m(eligible));
r=risk(eligible); idx=cluster(eligible); G=max(g);
O=sparse(g(r),idx(r),1,G,n);
F=sparse(g(~r),idx(~r),1,G,n);
end

function [out,brackets]=curve_eval(O,F,R,U,w,target)
H=[zeros(1,size(w,2));cumsum(O*w,1)]; N=[zeros(1,size(w,2));cumsum(F*w,1)];
r=R*w; n=U*w;
out=nan(size(w,2),numel(target)); brackets=strings(size(out));
for k=1:numel(n)
    if r(k)==0 || n(k)==0,continue;end
    for j=1:numel(target)
        ideal=n(k)*target(j);
        b=find(N(:,k)>=ideal-1e-10,1);
        if isempty(b),continue;end
        if abs(N(b,k)-ideal)<=1e-10
            out(k,j)=H(b,k)/r(k);
            if size(w,2)==1,brackets(k,j)=sprintf('%d/%d',N(b,k),n(k));end
        elseif b>1
            a=find(N(:,k)==N(b-1,k),1);
            x=(ideal-N(a,k))/(N(b,k)-N(a,k));
            out(k,j)=((1-x)*H(a,k)+x*H(b,k))/r(k);
            if size(w,2)==1,brackets(k,j)=sprintf('%d:%d/%d',N(a,k),N(b,k),n(k));end
        end
    end
end
end

function labels_19_20(A,W,NC,NS,p,out)
% 针对本轮校准、确认已存参数确定性回放原RK4。要求逐窗20 s标签
% 和首次越限都与冻结账本一致；失败即停止，不使用替代标签覆盖归档。
rows=cell(0,7); detail=cell(0,6);
tables={A,W}; params={NC,NS}; stages=["calibration","test"];
for i=1:2
    t=tables{i}; P=params{i}; x=P{:,{'H','D','R','Tg','d'}};
    f=rolling_rk4(x);
    below=f<-.005; first=ones(height(P),1)*Inf;
    [~,ix]=max(below,[],2); ok=any(below,2); first(ok)=(ix(ok)-1)*.02;
    assert(isequaln(first,P.Crossing_s));
    f=50*(1+f);
    for w=p.windows(:).'
        key=t(t.Condition=="clean"&abs(t.Window_s-w*.02)<1e-12,:);
        label20=min(f(:,w+2:end),[],2);
        label19=min(f(:,w+2:951),[],2);
        a=max(abs(label20-key.Target_Hz));
        assert(a<3e-11,'回放与冻结20 s标签不符');
        delta=label19-label20; changed=delta>3e-11;
        rows(end+1,:)={stages(i),w*.02,height(P),sum(changed),max(delta),a,19}; %#ok<AGROW>
        for j=find(changed).'
            detail(end+1,:)={stages(i),w*.02,P.Event(j),label19(j),label20(j),delta(j)}; %#ok<AGROW>
        end
    end
end
writetable(cell2table(rows,'VariableNames',{'Split','Window_s','Events','Changed','MaxDecrease_Hz','FrozenReplayMaxError_Hz','ShortHorizon_s'}),fullfile(out,'标签19至20秒核对.csv'));
writetable(cell2table(detail,'VariableNames',{'Split','Window_s','Event','Label19_Hz','Label20_Hz','Decrease_Hz'}),fullfile(out,'标签改变的事件.csv'));
end

function f=rolling_rk4(p)
h=p(:,1); damp=p(:,2); r=p(:,3); tg=p(:,4); loss=p(:,5);
f=zeros(numel(h),1001); z=zeros(numel(h),2);
rhs=@(y)[(-loss-damp.*y(:,1)+y(:,2))./(2*h),(min(.1,max(0,-y(:,1)./r))-y(:,2))./tg];
for j=1:4001-1
    a=rhs(z); b=rhs(z+.0025*a); c=rhs(z+.0025*b); d=rhs(z+.005*c);
    z=z+(.005/6)*(a+2*b+2*c+d);
    if mod(j,4)==0,f(:,j/4+1)=z(:,1);end
end
end

function y=boolfield(x)
% Pandas导出True/False，兼容MATLAB不同版本的读表推断。
if islogical(x)||isnumeric(x),y=logical(x);else,y=strcmpi(string(x),"true");end
end
