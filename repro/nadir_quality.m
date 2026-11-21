function nadir_quality(out)
% Generate archived protocol pair cases plus separately sampled single events.
% Simulator-only noise amplitudes and condition labels never enter select().
here = fileparts(mfilename('fullpath'));
project = fullfile(fileparts(here),'Project');
addpath(fullfile(project,'config'),fullfile(project,'core'),fullfile(project,'utils'));
c = jsondecode(fileread(fullfile(here,'nadir_quality_protocol.json')));
q = jsondecode(fileread(fullfile(out,'protocol.json')));
stage = jsondecode(fileread(fullfile(out,'stage.json')));
old = jsondecode(fileread(fullfile(here,'out','nadir-ecgr','confirmation-v1','freeze.json')));
prior = old.training_cubic_second_moment_hz2;
lambda = old.lambda;
assert(lambda==4);
map = estimator();
% Match the previously frozen ECGR implementation against cached development inputs.
source = fullfile(here,'out','revision');
a = load(fullfile(source,'trajectories.mat'),'grid');
vis = find(a.grid.t>=q.disturbance_s & a.grid.t<=q.disturbance_s+q.cutoff_relative_s);
archived = readmatrix(fullfile(here,'out','nadir-ecgr','confirmation-v1','development-features.csv'));
computed = features(a.grid.F(:,vis),map,lambda,prior);
assert(max(abs(computed(:,[10 8 9])-archived(1:size(computed,1),16:18)),[],'all')<1e-10, ...
    'ECGR estimator differs from the frozen implementation.');
writelines("PASS reproduced the historical frozen ECGR features.",fullfile(out,'matlab-anchor.log'));

nadir_pair_test('generate',out);
p = readtable(fullfile(out,'pair-index.csv'),TextType='string');
a = load(fullfile(out,'trajectories.mat'),'traceA','traceB','vis','post');
good = ismember(p.Status,["primary_eligible","true_tie","below_minimum_gap"]);
valid = p(good,:);
r = readtable(fullfile(out,'references.csv'));
na = height(r);
nb = height(p);
assert(all(isfinite(a.traceA(unique(valid.ReferenceID),a.vis)),'all'));
% Entity IDs distinguish reference members, candidates, and independently
% sampled single-test events. A shared reference has one observed window.
A = unique(valid.ReferenceID);
params = {'H','D','R','Tg','d'};
candidate = {'CandidateH','CandidateD','CandidateR','CandidateTg','CandidateD_pu'};
labelsA = min(a.traceA(A,a.post),[],2);
labelsB = min(a.traceB(valid.PairID,a.post),[],2);
writetable(table(valid.PairID,valid.ReferenceID,valid.RBand,valid.ReferenceID,na+valid.PairID, ...
    valid.Status,valid.DeltaTarget_Hz,VariableNames={'PairID','ReferenceID','RBand','A','B','Status','DeltaTarget_Hz'}), ...
    fullfile(out,'pair-keys.csv'));
[singles,f] = simulate_singles(q,stage,na+nb);
writetable(singles,fullfile(out,'single-index.csv'));
singleOK = singles.Status=="ok";
ids = [A;na+valid.PairID;singles.Entity(singleOK)];
truth = [labelsA;labelsB;singles.Target_Hz(singleOK)];
band = [r.RBand(A);valid.RBand;singles.RBand(singleOK)];
kinds = [repmat("reference",numel(A),1);repmat("candidate",height(valid),1); ...
         repmat("single",nnz(singleOK),1)];
parameters = [r{A,params};valid{:,candidate};singles{singleOK,params}];
entity = [table(ids,kinds,band,truth,VariableNames={'Entity','Kind','RBand','Target_Hz'}), ...
          array2table(parameters,VariableNames=params)];
writetable(entity,fullfile(out,'entity-index.csv'));
clean = [a.traceA(A,a.vis);a.traceB(valid.PairID,a.vis);f(singleOK,:)];
total = height(entity)*(1+(numel(c.conditions)-1)*stage.repeats);
featuresOut = zeros(total,13);
keys = zeros(total,3);
cursor = 0;
for i=1:height(entity)
    trend = clean(i,:);
    % Simulation noise is scaled against the unavailable clean trajectory.
    % This scaling quantity is never exported into the gate feature columns.
    ref = sqrt(mean((trend-50).^2));
    stream = RandStream('mt19937ar','Seed',stage.noise_seed + ids(i));
    z = randn(stream,stage.repeats,c.pre_samples+26);
    ar = z;
    for j=2:size(z,2)
        ar(:,j)=c.rho*ar(:,j-1)+sqrt(1-c.rho^2)*z(:,j);
    end
    for condition=1:numel(c.conditions)
        switch string(c.conditions{condition})
            case "clean"
                pre = 50*ones(1,c.pre_samples);
                post = trend;
                rep=0;
            otherwise
                switch string(c.conditions{condition})
                    case "white40", sigma=ref/100; noise=z;
                    case "white30", sigma=ref*10^(-30/20); noise=z;
                    case "white20", sigma=ref/10; noise=z;
                    case "ar30", sigma=ref*10^(-30/20); noise=ar;
                    case "jump"
                        sigma=[repmat(ref*10^(-c.jump_pre_snr_db/20),1,c.pre_samples), ...
                               repmat(ref*10^(-c.jump_post_snr_db/20),1,26)];
                        % At the disturbance, both variance and correlation change.
                        noise=z;
                        noise(:,c.pre_samples+1)=z(:,c.pre_samples+1);
                        for j=c.pre_samples+2:size(noise,2)
                            noise(:,j)=c.rho*noise(:,j-1)+sqrt(1-c.rho^2)*z(:,j);
                        end
                end
                d = noise.*sigma;
                pre = 50+d(:,1:c.pre_samples);
                post = trend+d(:,c.pre_samples+1:end);
                rep=(1:stage.repeats).';
        end
        [preQ,windowQ,hybridQ]=quality(pre,post,c.signal_floor_hz);
        X=features(post,map,lambda,prior);
        span=cursor+(1:size(X,1));
        keys(span,:)=[repmat([ids(i),condition],numel(span),1),rep];
        featuresOut(span,:)=[preQ,windowQ,hybridQ,X];
        cursor=span(end);
    end
end
assert(cursor==total && all(isfinite(featuresOut),'all'));
columns={'Entity','Condition','Rep','PreQuality','WindowQuality','HybridQuality','Endpoint','InitialRate', ...
    'Linear','Curvature','Slope1','Slope2','Slope3','Symmetric','Antisymmetric','ECGREndpoint'};
writetable(array2table([keys featuresOut],VariableNames=columns),fullfile(out,'visible-features.csv'));
end

function [p,w,h] = quality(pre,post,floorHz)
% Use the demeaned pre-event scale and a third-difference current-window
% scale. The latter is divided by sqrt(20), its white-noise gain.
preScale=sqrt(mean((pre-mean(pre,2)).^2,2));
signal=max(sqrt(mean((post-mean(pre,2)).^2,2)),floorHz);
p=preScale./signal;
w=sqrt(mean(diff(post,3,2).^2,2))./sqrt(20)./signal;
h=max(p,w);
end

function op=estimator()
% Kept in lockstep with the archived ECGR implementation. Historical
% equivalence is checked before this stage is allowed to simulate.
u=linspace(0,1,26).';
v=u(2:end-1);
L=[-(1-v),eye(24),-v];
B=[v.*(1-v),v.*(1-v).*(2*v-1)];
C=chol(L*(.5*toeplitz(.65.^(0:25))+.5*eye(26))*L','lower');
Z=C\B; W=C\L;
op.g=Z'*Z;
op.ls=op.g\(Z'*W);
op.rough=W-Z*op.ls;
end

function X=features(f,op,lambda,prior)
% Original four fixed inputs and the archived ECGR estimator.
t=(0:25)*.02;
u=t(:)/t(end);
z=f-50;
beta=[ones(26,1),u,u.^2]\z.';
edge=[1 9 17 25];
d=zeros(size(f,1),3);
for k=1:3
    i=edge(k):edge(k+1);
    tc=t(i)-mean(t(i));
    d(:,k)=z(:,i)*tc.'/(tc*tc.');
end
b0=f*op.ls.';
proxy=sum((f*op.rough.').^2,2)/22;
g=op.g;
z0=b0*g;
p=lambda*proxy/prior;
determinant=g(1,1)*(g(2,2)+p)-g(1,2)^2;
b1=(z0(:,1).*(g(2,2)+p)-z0(:,2)*g(1,2))./determinant;
b2=(z0(:,2)*g(1,1)-z0(:,1)*g(1,2))./determinant;
X=[z(:,26),(z(:,6)-z(:,1))/.10,beta(2:3,:).',d,b1,b2,z(:,26)];
end

function [index,windows]=simulate_singles(q,stage,offset)
    rng(q.reference_seed+1,'twister');
    n=stage.single_per_band*size(q.r_bands,1);
    band=repelem((1:size(q.r_bands,1)).',stage.single_per_band);
    v=rand(n,5);
    % Wider original parametric ranges than the directional pairing anchors.
    ranges=[2 6;.5 2;.04 .06;.2 1;.02 .08];
    p=ranges(:,1).'+v.*diff(ranges,1,2).';
    p(:,3)=q.r_bands(band,1)+v(:,3).*diff(q.r_bands(band,:),1,2);
    ids=offset+(1:n).';
    states=repmat("ok",n,1);
    details=strings(n,1);
    targets=nan(n,1);
    windows=nan(n,26);
    for i=1:n
        sys=sys_config(); sim=sim_config();
        sys.H=p(i,1); sys.D=p(i,2); sys.R=p(i,3); sys.Tg=p(i,4); sim.dP=p(i,5);
        try
            [t,f]=simulate_freq(sys,sim);
            vis=t>=1 & t<=1.5; after=t>1.5;
            assert(nnz(vis)==26 && nnz(after)==925 && all(isfinite(f)));
            windows(i,:)=f(vis);
            targets(i)=min(f(after));
        catch failure
            states(i)="simulation_failure";
            details(i)=string(failure.identifier)+" "+string(failure.message);
        end
    end
    index=[table(ids,band,states,details,targets,VariableNames={'Entity','RBand','Status','Detail','Target_Hz'}), ...
           array2table(p,VariableNames={'H','D','R','Tg','d'})];
end
