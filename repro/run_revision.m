function run_revision()
% Rebuild the revised manuscript experiments without overwriting legacy data.
here = fileparts(mfilename('fullpath'));
project = fullfile(fileparts(here), 'Project');
addpath(fullfile(project,'config'), fullfile(project,'core'), fullfile(project,'utils'));
out = fullfile(here,'out','revision');
if ~isfolder(out), mkdir(out); end
sys = sys_config();
sim = sim_config();
grid = gen_dataset(sys, sim, param_space('full'));
t = grid.t;
cut = sim.t_dist + sim.tau_obs;
visible = t >= sim.t_dist & t <= cut;
assert(nnz(visible) == 26 && all(grid.F(:,t == sim.t_dist) == sys.f_nom));
[g, lead] = event_group(grid.F,t,sim);
assert(isequal(g ~= "after",grid.posthoc));
y = min(grid.F(:,t > cut),[],2);
assert(max(abs(y-grid.nadir)) < 1e-10);

eligible = find(g == "after");
rng(42,'twister');
perm = eligible(randperm(numel(eligible)));
nTrain = floor(.6*numel(perm));
nVal = floor(.2*numel(perm));
tr = perm(1:nTrain);
va = perm(nTrain+1:nTrain+nVal);
te = perm(nTrain+nVal+1:end);
allTest = [te; find(g ~= "after")];
split = repmat("excluded-from-conditional",numel(y),1);
split(tr) = "train"; split(va) = "validation"; split(te) = "test";
gridIDs = (1:numel(y)).';
index = table(gridIDs,grid.params(:,1),grid.params(:,2),grid.params(:,3),grid.params(:,4), ...
    grid.params(:,5),g,lead,y,split,ismember(gridIDs,allTest), ...
    'VariableNames',{'ID','H','D','R','Tg','d','Event','Lead_s','Target_Hz','Split','InFullTest'});

names = ["endpoint","initial_rate","quadratic","segmented","fitted_segments","vmd"];
X = features(grid.F(:,visible),t(visible),sys.f_nom);
config = struct('release',version('-release'),'matlab',version,'cut_s',cut, ...
    'split_seed',42,'continuous_seed',777,'noise_seed',1681,'noise_repeats',10, ...
    'tree_cycles',100,'tree_learn_rate',0.1,'tree_max_depth',3,'vmd_modes',4, ...
    'threshold_Hz',49.75,'cascade_fast_fraction',.25);
rng(777,'twister');
limits = [2 6; .5 2; .04 .06; .2 1; .02 .08];
params = limits(:,1).' + rand(500,5).*diff(limits,1,2).';
F = zeros(500,numel(t));
Fshift = F;
for i = 1:size(F,1)
    s = sys;
    s.H = params(i,1); s.D = params(i,2); s.R = params(i,3); s.Tg = params(i,4);
    sm = sim; sm.dP = params(i,5);
    [~,F(i,:)] = simulate_freq(s,sm);
    Fshift(i,:) = cascade_frequency(t,s,sm,config.cascade_fast_fraction);
end
[gc, lc] = event_group(F,t,sim);
yc = min(F(:,t > cut),[],2);
Xc = features(F(:,visible),t(visible),sys.f_nom);
[gs, ls] = event_group(Fshift,t,sim);
ys = min(Fshift(:,t > cut),[],2);
Xs = features(Fshift(:,visible),t(visible),sys.f_nom);
continuousIDs = (1:numel(yc)).';
contin = table(continuousIDs,params(:,1),params(:,2),params(:,3),params(:,4), ...
    params(:,5),gc,lc,yc,'VariableNames',index.Properties.VariableNames(1:9));
shift = contin;
shift.Event = gs;
shift.Lead_s = ls;
shift.Target_Hz = ys;

window = zeros(numel(y),3);
widths = [.10 .25 .50];
for k = 1:numel(widths)
    last = find(t <= sim.t_dist+widths(k),1,'last');
    window(:,k) = grid.F(:,last) - sys.f_nom;
end
rng(config.noise_seed,'twister');
fw = grid.F(te,visible);
power = mean((fw-sys.f_nom).^2,2);
snrs = [20 30 40];
noiseFeatures = cell(config.noise_repeats,numel(snrs),numel(names));
for rep = 1:config.noise_repeats
    noise = randn(size(fw));
    for j = 1:numel(snrs)
        x = features(fw + noise.*sqrt(power/10^(snrs(j)/10)),t(visible),sys.f_nom);
        noiseFeatures(rep,j,:) = reshape(x,1,1,[]);
    end
end
writetable(index,fullfile(out,'grid-index.csv'));
writetable(contin,fullfile(out,'continuous-index.csv'));
writetable(shift,fullfile(out,'cascade-index.csv'));
save(fullfile(out,'features.mat'),'X','Xc','Xs','noiseFeatures','te','window','widths','snrs');
save(fullfile(out,'trajectories.mat'),'grid','F','Fshift','t','params','config');
fid = fopen(fullfile(out,'config.json'),'w');
assert(fid >= 0);
fprintf(fid,'%s\n',jsonencode(config,PrettyPrint=true));
fclose(fid);
fprintf('Simulation and feature export complete. Training %d, validation %d, conditional test %d, full test %d.\n', ...
    numel(tr),numel(va),numel(te),numel(allTest));
end

function f = cascade_frequency(t,s,sim,fraction)
after = t > sim.t_dist;
rhs = @(~,state) [(-sim.dP-s.D*state(1)+state(3))/(2*s.H); ...
    (pfr_model(state(1),s)-state(2))/(fraction*s.Tg); ...
    (state(2)-state(3))/((1-fraction)*s.Tg)];
[~,state] = ode45(rhs,[sim.t_dist,t(after)],[0;0;0],odeset(RelTol=sim.rel_tol));
f = s.f_nom*ones(size(t));
f(after) = s.f_nom*(1+state(2:end,1)).';
end

function [group, lead] = event_group(F,t,sim)
post = find(t > sim.t_dist);
[~,i] = min(F(:,post),[],2);
lead = t(post(i)).' - (sim.t_dist+sim.tau_obs);
group = repmat("at",size(lead));
group(lead < 0) = "before";
group(lead > 0) = "after";
end

function result = features(F,t,f0)
x = F-f0;
edge = [1 9 17 25];
seg = zeros(size(x,1),3);
fit = seg;
for j = 1:3
    idx = edge(j):edge(j+1);
    s = t(idx)-mean(t(idx));
    seg(:,j) = (x(:,idx(end))-x(:,idx(1)))/(t(idx(end))-t(idx(1)));
    fit(:,j) = (x(:,idx)*s.')/sum(s.^2);
end
u = (t-t(1)).'/(t(end)-t(1));
c = [ones(numel(t),1),u,u.^2]\x.';
endpt = x(:,end);
rate = (x(:,6)-x(:,1))/(t(6)-t(1));
energy = zeros(size(x,1),5);
for i = 1:size(x,1)
    signal = x(i,:).' - mean(x(i,:));
    imf = vmd(signal,NumIMF=4,AbsoluteTolerance=1e-6);
    e = sum(imf.^2,1);
    assert(numel(e)==4 && all(isfinite(e)) && sum(e)>0,'Invalid VMD energy.');
    p = e/sum(e);
    energy(i,:) = [p,-sum(p.*log2(max(p,eps)))];
end
result = {endpt,[endpt rate],[endpt c(2:3,:).'],[endpt seg],[endpt fit],[endpt energy]};
end

