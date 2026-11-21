function nadir_pair_test(stage, runDir)
% Independent fixed-cutoff study. Stages intentionally separate inputs, inference,
% and labels. Use the PowerShell driver to preserve logs and enforce hashes.
here = fileparts(mfilename('fullpath'));
src = fullfile(here,'out','revision');
project = fullfile(fileparts(here),'Project');
addpath(fullfile(project,'config'),fullfile(project,'core'),fullfile(project,'utils'));
if stage == "anchor"
    c = jsondecode(fileread(fullfile(here,'nadir_pair_protocol.json')));
else
    c = jsondecode(fileread(fullfile(runDir,'protocol.json')));
end
switch string(stage)
    case "anchor"
        historical_anchor(src,runDir,c);
    case "generate"
        generate(src,runDir,c);
    case "summarize"
        summarize(runDir,c);
    otherwise
        error('Unknown stage %s.',string(stage));
end
end

function historical_anchor(src,out,c)
% Extract 26-point features and exact suffix labels from the saved baseline.
a = load(fullfile(src,'trajectories.mat'),'grid');
g = readtable(fullfile(src,'grid-index.csv'),TextType='string');
[t,vis,post] = indices(c);
assert(max(abs(a.grid.t(:)-t(:))) < 1e-12 && height(g)==size(a.grid.F,1));
assert(max(abs(min(a.grid.F(:,post),[],2)-g.Target_Hz)) < 1e-10);
test = g.Split=="test";
features = array2table(extract(a.grid.F(test,vis),c),VariableNames=feature_names());
features = addvars(features,g.ID(test),Before=1,NewVariableNames='ID');
writetable(features,fullfile(out,'anchor-features.csv'));
writelines([string(version); string(computer)],fullfile(out,'matlab-environment.txt'));
end

function [t,vis,post] = indices(c)
% Index by integer sample counts: t_c itself is visible and excluded from y.
t = (0:round(c.end_s/c.step_s))*c.step_s;
start = round(c.disturbance_s/c.step_s)+1;
cut = start+round(c.cutoff_relative_s/c.step_s);
vis = start:cut;
post = cut+1:numel(t);
assert(numel(vis)==26 && abs(t(post(1))-1.52)<1e-12);
end

function names = feature_names()
names = {'Endpoint','InitialRate','Linear','Curvature','Slope1','Slope2','Slope3'};
end

function X = extract(f,c)
% Formulas follow run_revision.m. Slopes use samples through 0.48 s;
% the untouched 0.50 s sample enters the endpoint feature for every model.
x = f-c.nominal_hz;
t = (0:25)*c.step_s;
u = t(:)/t(end);
beta = [ones(size(u)),u,u.^2]\x.';
edge = [1 9 17 25];
slopes = zeros(size(x,1),3);
for j = 1:3
    k = edge(j):edge(j+1);
    time = t(k)-mean(t(k));
    slopes(:,j) = x(:,k)*time.'/sum(time.^2);
end
X = [x(:,26),(x(:,6)-x(:,1))/t(6),beta(2:3,:).',slopes];
end

function generate(src,out,c)
assert(~isfile(fullfile(out,'pair-index.csv')),'Preserve existing pair ledger.');
[t,vis,post] = indices(c);
old = readtable(fullfile(src,'grid-index.csv'));
continuum = readtable(fullfile(src,'continuous-index.csv'));
columns = {'H','D','R','Tg','d'};
interval = readtable(fullfile(fileparts(src),'nadir-intervals','scenarios.csv'),TextType='string');
oldMatch = readtable(fullfile(fileparts(src),'nadir-identifiability','matched-scan.csv'),TextType='string');
oldAnchor = [repmat([4 1 .05 .5],height(oldMatch),1),oldMatch.ReferenceDeficit];
oldCandidate = oldAnchor;
for k = 1:height(oldMatch)
    column = find(strcmp(columns,oldMatch.Factor(k)));
    oldCandidate(k,column) = oldMatch.FactorValue(k);
end
oldCandidate(:,5) = oldMatch.MatchedDeficit;
previous = [old{:,columns}; continuum{:,columns}; interval{:,{'H','D','R','Tg','Deficit'}}; ...
    oldAnchor; oldCandidate(all(isfinite(oldCandidate),2),:)];
assert(~any(any(old.R >= c.r_bands(:,1).' & old.R <= c.r_bands(:,2).',2)), ...
    'A frozen grid R level intersects a held-out band.');
N = c.references_per_r_band*size(c.r_bands,1);
rng(c.reference_seed,'twister');
u = rand(N,5);
r = c.reference_ranges_H_D_Tg_d;
para = zeros(N,5);
para(:,[1 2 4 5]) = r(:,1).' + u(:,[1 2 4 5]).*diff(r,1,2).';
bands = repelem((1:size(c.r_bands,1)).',c.references_per_r_band);
para(:,3) = c.r_bands(bands,1)+u(:,3).*diff(c.r_bands(bands,:),1,2);
factor = string(c.candidate_factors);
count = N*numel(factor);
ratios = c.candidate_multipliers;
coord = [1 2 4];
reference = array2table(para,VariableNames=columns);
reference = addvars(reference,(1:N).',bands,Before=1,NewVariableNames={'ReferenceID','RBand'});
writetable(reference,fullfile(out,'references.csv'));

% Candidate IDs are fixed before solving. Failed bracket/solver calls are kept
% with signed bracket residuals (when available), and no replacements are drawn.
pairs = reference(repelem((1:N).',numel(factor)),:);
pairs = addvars(pairs,(1:count).',repmat(factor(:),N,1),Before=1, ...
    NewVariableNames={'PairID','ChangedFactor'});
pairs = [pairs,array2table(nan(count,12),VariableNames= ...
    {'CandidateH','CandidateD','CandidateR','CandidateTg','CandidateD_pu', ...
    'ResidualLow_Hz','ResidualHigh_Hz','EndpointA_Hz','EndpointB_Hz', ...
    'TargetA_Hz','TargetB_Hz','DeltaTarget_Hz'})];
pairs.Status = repmat("pending",count,1);
pairs.Detail = strings(count,1);
pairs.EventA = strings(count,1);
pairs.EventB = strings(count,1);
traceA = nan(N,numel(t));
traceB = nan(count,numel(t));
xRows = (2*(1+c.noise_repeats))*count;
X = nan(xRows,7);
keys = nan(xRows,4);
filled = 0;
for i = 1:N
    base = para(i,:);
    assert(all(max(abs(previous-base),[],2)>1e-9),'Historical reference collision.');
    reason = "";
    try
        a = response(base,t,c);
        traceA(i,:) = a;
    catch exception
        reason = string(exception.identifier) + " " + string(exception.message);
    end
    for k = 1:numel(factor)
        id = (i-1)*numel(factor)+k;
        bPar = base;
        bPar(coord(k)) = bPar(coord(k))*ratios(k);
        bPar(5) = NaN;
        pairs{id,{'CandidateH','CandidateD','CandidateR','CandidateTg','CandidateD_pu'}} = bPar;
        if reason~=""
            pairs.Status(id) = "reference_simulation_failure";
            pairs.Detail(id) = reason;
            continue
        end
        ea = a(vis(end));
        ya = min(a(post));
        pairs.EndpointA_Hz(id) = ea;
        pairs.TargetA_Hz(id) = ya;
        pairs.EventA(id) = event_type(a,vis);
        stage = "bracket_simulation_failure";
        try
            mismatch = @(d) endpoint_gap(bPar,d,ea,c,t,vis(end));
            bracket = c.deficit_bracket(:).';
            low = mismatch(bracket(1));
            high = mismatch(bracket(2));
            pairs.ResidualLow_Hz(id) = low;
            pairs.ResidualHigh_Hz(id) = high;
            if low==0
                d = bracket(1);
            elseif high==0
                d = bracket(2);
            elseif sign(low)==sign(high)
                pairs.Status(id) = "no_bracket";
                continue
            else
                stage = "root_failure";
                [d,~,exitFlag] = fzero(mismatch,bracket,optimset('TolX',c.root_tolerance_pu));
                assert(exitFlag==1 && d>=bracket(1) && d<=bracket(2),'Root did not converge in bracket.');
            end
            bPar(5) = d;
            pairs.CandidateD_pu(id) = d;
            if any(max(abs(previous-bPar),[],2)<=1e-9)
                pairs.Status(id) = "historical_collision";
                continue
            end
            stage = "candidate_simulation_failure";
            b = response(bPar,t,c);
            traceB(id,:) = b;
            pairs.EndpointB_Hz(id) = b(vis(end));
            pairs.TargetB_Hz(id) = min(b(post));
            pairs.EventB(id) = event_type(b,vis);
            dy = pairs.TargetB_Hz(id)-ya;
            pairs.DeltaTarget_Hz(id) = dy;
            if abs(pairs.EndpointB_Hz(id)-ea)>c.endpoint_tolerance_hz
                pairs.Status(id) = "outside_endpoint_tolerance";
                continue
            end
            % Magnitude eligibility never determines inclusion in the direction
            % analysis. Genuine ties and small nonzero gaps stay in that set.
            if abs(dy)<=c.tie_hz
                pairs.Status(id) = "true_tie";
            elseif abs(dy)<c.minimum_target_gap_hz
                pairs.Status(id) = "below_minimum_gap";
            else
                pairs.Status(id) = "primary_eligible";
            end
            stage = "feature_failure";
            observed = [a(vis);b(vis)];
            % Reference uses identical noise across all its child comparisons.
            % Distinct members have independent streams and draw matrices.
            streamA = RandStream('mt19937ar','Seed',c.noise_seed+i);
            streamB = RandStream('mt19937ar','Seed',c.noise_seed+N+id);
            noiseA = randn(streamA,c.noise_repeats,26);
            noiseB = randn(streamB,c.noise_repeats,26);
            sigma = sqrt(mean((observed-c.nominal_hz).^2,2))*10^(-c.white_snr_db/20);
            block = extract([observed; observed(1,:)+sigma(1)*noiseA; ...
                observed(2,:)+sigma(2)*noiseB],c);
            assert(all(isfinite(block),'all'),'Nonfinite model inputs.');
            R = (1:size(block,1))+filled;
            X(R,:) = block;
            keys(R,:) = [repmat([i id],numel(R),1), ...
                [1;2;ones(c.noise_repeats,1);2*ones(c.noise_repeats,1)], ...
                [0;0;(1:c.noise_repeats).';(1:c.noise_repeats).']];
            filled = R(end);
        catch exception
            pairs.Status(id) = stage;
            pairs.Detail(id) = string(exception.identifier) + " " + string(exception.message);
        end
    end
end
keys = keys(1:filled,:);
x = array2table(X(1:filled,:),VariableNames=feature_names());
x = addvars(x,keys(:,1),keys(:,2),keys(:,3),keys(:,4),Before=1, ...
    NewVariableNames={'ReferenceID','PairID','Member','Rep'});
x.Condition = repmat("white30",height(x),1);
x.Condition(x.Rep==0) = "clean";
writetable(pairs,fullfile(out,'pair-index.csv'));
writetable(pairs(pairs.Status~="primary_eligible",:),fullfile(out,'excluded-and-failed.csv'));
writetable(x,fullfile(out,'features.csv'));
save(fullfile(out,'trajectories.mat'),'traceA','traceB','t','vis','post');
end

function z = endpoint_gap(par,d,endpoint,c,t,cut)
% Solve the endpoint using the same full integration routine as the final
% candidate. This avoids a solver-horizon difference at the matching step.
par(5) = d;
f = response(par,t,c);
z = f(cut)-endpoint;
end

function f = response(par,t,c)
sys = sys_config();
sim = sim_config();
assert(sys.f_nom==c.nominal_hz && sys.p_max==.1 && sim.rel_tol==1e-8);
sys.H = par(1); sys.D = par(2); sys.R = par(3); sys.Tg = par(4);
sim.dP = par(5);
assert(sim.t_end==c.end_s && sim.dt==c.step_s && sim.t_dist==c.disturbance_s ...
    && sim.tau_obs==c.cutoff_relative_s);
[time,f] = simulate_freq(sys,sim);
assert(numel(time)==numel(t) && max(abs(time-t))<1e-12 && all(isfinite(f)), ...
    'Incomplete or nonfinite simulation.');
end

function name = event_type(f,visible)
[~,k] = min(f(visible(1):end));
j = k+visible(1)-1;
types = ["before","at","after"];
name = types(sign(j-visible(end))+2);
end

function summarize(out,c)
% Transform endpoint estimates into within-pair contrasts. All matched pairs
% enter the direction denominator; the amplitude threshold defines primary
% error analysis only. Replicates are nested inside their reference event.
p = readtable(fullfile(out,'pair-index.csv'),TextType='string');
ref = readtable(fullfile(out,'references.csv'));
score = readtable(fullfile(out,'predictions.csv'),TextType='string');
names = string(c.features);
accepted = ismember(p.Status,["primary_eligible","true_tie","below_minimum_gap"]);
failures = groupsummary(p,["ChangedFactor","RBand","Status"]);
failures.Attempted = repmat(height(p),height(failures),1);
writetable(failures,fullfile(out,'status-counts.csv'));
factors = ["all"; unique(p.ChangedFactor)];
counts = nan(numel(factors),11);
for j = 1:numel(factors)
    use = factors(j)=="all" | p.ChangedFactor==factors(j);
    attempted = nnz(use);
    matched = nnz(use&accepted);
    featureFailures = nnz(use & p.Status=="feature_failure");
    % A feature failure is downstream of a successful geometry match.
    geometric = matched+featureFailures;
    counts(j,:) = [attempted,geometric,attempted-geometric,(attempted-geometric)/attempted, ...
        matched,nnz(use&p.Status=="primary_eligible"),nnz(use&p.Status=="true_tie"), ...
        nnz(use&p.Status=="below_minimum_gap"),featureFailures, ...
        nnz(use&p.Status=="no_bracket"),nnz(use&p.Status=="outside_endpoint_tolerance")];
end
flow = array2table(counts,VariableNames={'Attempted','GeometricMatches','MatchFailures', ...
    'MatchFailureRate','ScorableMatches','PrimaryPairs','TrueTies','SmallNonTies', ...
    'FeatureFailures','NoBracket','OutsideTolerance'});
writetable(addvars(flow,factors,Before=1,NewVariableNames='ChangedFactor'),fullfile(out,'pair-flow.csv'));

selected = p(accepted,:);
n = height(selected);
assert(height(score)==n*2*(1+c.noise_repeats)*numel(names),'Incomplete prediction ledger.');
groups = ["clean","white30"];
pairBlocks = cell(numel(groups)*numel(names),1);
rows = cell(numel(groups)*numel(names),1);
contrastBlocks = cell(numel(groups)*5,1);
ix = 0;
cx = 0;
rng(c.bootstrap_seed,'twister');
B = c.bootstrap_draws;
% Resample reference clusters within the two prespecified R bands, including
% references with no scorable or primary pair. Keep every child and repeat.
multiplicity = zeros(height(ref),B);
for band = unique(ref.RBand).'
    ids = ref.ReferenceID(ref.RBand==band);
    draw = ids(randi(numel(ids),numel(ids),B));
    for id = ids.'
        multiplicity(id,:) = sum(draw==id,1);
    end
end
truth = selected.DeltaTarget_Hz;
primary = selected.Status=="primary_eligible";
yd = direction(truth,c.tie_hz);
for condition = groups
    reps = 0;
    if condition=="white30", reps = 1:c.noise_repeats; end
    truthBlock = repmat(truth,1,numel(reps));
    eligible = repmat(primary,1,numel(reps));
    truthDirection = repmat(yd,1,numel(reps));
    baseline = [];
    initial = [];
    for name = names.'
        ix = ix+1;
        delta = nan(n,numel(reps));
        ahat = delta;
        bhat = delta;
        for j = 1:numel(reps)
            use = score(score.Condition==condition & score.Rep==reps(j) & score.Feature==name,:);
            first = sortrows(use(use.Member==1,:),'PairID');
            second = sortrows(use(use.Member==2,:),'PairID');
            assert(isequal(first.PairID,selected.PairID) && isequal(second.PairID,selected.PairID));
            assert(isequal(first.ReferenceID,selected.ReferenceID) && isequal(second.ReferenceID,selected.ReferenceID));
            ahat(:,j) = first.Prediction_Hz;
            bhat(:,j) = second.Prediction_Hz;
            delta(:,j) = bhat(:,j)-ahat(:,j);
        end
        predDir = direction(delta,c.tie_hz);
        err = abs(delta-truthBlock);
        correct = predDir==truthDirection;
        N = numel(err);
        allErrors = err(eligible)*1000;
        stats = [NaN NaN NaN];
        if ~isempty(allErrors)
            stats = [mean(allErrors),sqrt(mean(allErrors.^2)),linear_quantile(allErrors,.95)];
        end
        rows{ix} = table(condition,name,n,nnz(primary),N,numel(allErrors), ...
            stats(1),stats(2),stats(3),nnz(correct),nnz(correct)/N, ...
            nnz(predDir==0),nnz(truthDirection==0),nnz(yd==0), ...
            nnz(correct&eligible)/nnz(eligible), ...
            'VariableNames',{'Condition','Feature','ScorablePairs','PrimaryPairs','DirectionTrials', ...
            'ErrorTrials','DeltaMAE_mHz','DeltaRMSE_mHz','DeltaP95_mHz','CorrectDirections', ...
            'DirectionAccuracy','PredictedTieTrials','TrueTieTrials','TrueTiePairs','PrimaryDirectionAccuracy'});
        pairKeys = repmat(selected(:,{'PairID','ReferenceID','RBand','ChangedFactor','Status'}),numel(reps),1);
        pid = repmat(reps,n,1);
        pairBlocks{ix} = [pairKeys,table(repmat(condition,N,1),repmat(name,N,1),pid(:), ...
            ahat(:),bhat(:),truthBlock(:),delta(:),err(:),truthDirection(:),predDir(:),correct(:), ...
            'VariableNames',{'Condition','Feature','Rep','PredictionA_Hz','PredictionB_Hz', ...
            'DeltaTarget_Hz','DeltaPrediction_Hz','AbsDeltaError_Hz','TrueDirection','PredictedDirection','DirectionCorrect'})];
        if name=="endpoint"
            baseline = err;
            continue
        end
        % Negative change favors the feature against the named comparator.
        cx = cx+1;
        contrastBlocks{cx} = summarize_contrast(condition,name,"endpoint",err,baseline, ...
            primary,selected.ReferenceID,multiplicity);
        if name=="initial_rate"
            initial = err;
        else
            cx = cx+1;
            contrastBlocks{cx} = summarize_contrast(condition,name,"initial_rate",err,initial, ...
                primary,selected.ReferenceID,multiplicity);
        end
    end
end
writetable(vertcat(pairBlocks{:}),fullfile(out,'pair-predictions.csv'));
writetable(vertcat(rows{:}),fullfile(out,'summary.csv'));
contrasts = vertcat(contrastBlocks{:});
writetable(contrasts(contrasts.Comparator=="endpoint",:),fullfile(out,'contrast-vs-endpoint.csv'));
writetable(contrasts(contrasts.Comparator=="initial_rate",:),fullfile(out,'contrast-vs-initial-rate.csv'));
end

function row = summarize_contrast(condition,name,against,err,base,primary,ids,weights)
% Average each pair across replicates, then sum pairs inside a reference.
d = 1000*mean(err(primary,:)-base(primary,:),2);
cluster = ids(primary);
shape = [size(weights,1) 1];
sums = accumarray(cluster,d,shape);
ns = accumarray(cluster,ones(size(cluster)),shape);
denom = ns.'*weights;
draws = (sums.'*weights)./denom;
draws = draws(isfinite(draws));
ci = [NaN NaN];
if nnz(ns)>=2 && ~isempty(draws)
    ci = linear_quantile(draws,[.025 .975]);
end
row = table(condition,name,against,nnz(primary),nnz(ns),mean(d),ci(1),ci(2),numel(draws), ...
    'VariableNames',{'Condition','Feature','Comparator','PrimaryPairs','ClustersWithPrimaryPairs', ...
    'ChangeMAE_mHz','Low95_mHz','High95_mHz','ValidBootstrapDraws'});
end

function out = direction(values,tolerance)
% -1 means B is lower, +1 means A is lower, 0 is a declared tie.
out = sign(values);
out(abs(values)<=tolerance) = 0;
end

function y = linear_quantile(x,p)
% Linear interpolation on rank 1+(n-1)p, no statistics toolbox required.
x = sort(x(:));
k = 1+(numel(x)-1)*p;
lo = floor(k);
hi = ceil(k);
a = reshape(x(lo),size(p));
b = reshape(x(hi),size(p));
y = a+(k-lo).*(b-a);
end
