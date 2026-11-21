function nadir_risk(stage,out)
% Fixed-cutoff risk study. Simulation truth is used only for labels and
% controlled noise generation. Calibration only receives causal features.
here=fileparts(mfilename('fullpath'));
project=fullfile(here,'..','Project');
addpath(fullfile(project,'config'),fullfile(project,'core'),fullfile(project,'utils'));
c=jsondecode(fileread(fullfile(out,'protocol.json')));
switch string(stage)
    case "development"
        file=fullfile(out,'development.csv');
        assert(~isfile(file),'Preserve the historical calibration packet.');
        t=[simulate(c.design_n,c.design_seed,0,"design",false,false); ...
           simulate(c.calibration_n,c.calibration_seed,c.design_n,"cal",false,false)];
    case "confirmation"
        file=fullfile(out,'confirmation.csv');
        assert(isfile(fullfile(out,'freeze.json')) && ~isfile(file),'Run requires a frozen protocol.');
        t=[simulate(c.final_n,c.final_seed,0,"final",false,true); ...
           simulate(c.extrapolation_n,c.extrapolation_seed,10000,"final",true,false); ...
           archived(c.archive_noise_seed,here)];
    otherwise
        error('Unknown stage.');
end
writetable(t,file);
fprintf('Wrote %d rows (%s).\n',height(t),stage);
end

function rows=simulate(n,seed,offset,split,extrap,fresh)
rng(seed,'twister');
limits=[2 6;.5 2;.04 .06;.2 1;.02 .08];
if extrap, limits(1,:)=[6.25 7.5]; end
p=limits(:,1).'+rand(n,5).*diff(limits,1,2).';
mixed=randi(3,n,1);
eps=randn(n,26);
rows=cell(n*(1+2*~extrap+fresh),15);
k=0;
for i=1:n
    s=sys_config(); sm=sim_config();
    q=p(i,:);
    s.H=q(1); s.D=q(2); s.R=q(3); s.Tg=q(4); sm.dP=q(5);
    [t,f]=simulate_freq(s,sm);
    name="mixed";
    if extrap, name="inertia_high"; end
    k=k+1;
    rows(k,:)=row(offset+i,split,name,mixed(i),q,t,f,eps(i,:));
    if ~extrap
        k=k+1;
        rows(k,:)=row(offset+i,split,"noise_jump",4,q,t,f,eps(i,:));
        k=k+1;
        rows(k,:)=row(offset+i,split,"ar20",5,q,t,f,eps(i,:));
    end
    if fresh
        visible=t>=sm.t_dist;
        dt=t(visible)-sm.t_dist;
        rhs=@(~,z)[(-sm.dP-s.D*z(1)+z(3))/(2*s.H); ...
           (pfr_model(z(1),s)-z(2))/(.25*s.Tg);(z(2)-z(3))/(.75*s.Tg)];
        [~,z]=ode45(rhs,dt,[0;0;0],odeset(RelTol=sm.rel_tol));
        g=f; g(visible)=50*(1+z(:,1)).';
        k=k+1;
        rows(k,:)=row(offset+i,split,"cascade_fresh",mixed(i),q,t,g,eps(i,:));
    end
end
assert(k==size(rows,1));
rows=as_table(rows);
end

function rows=archived(seed,here)
src=load(fullfile(here,'out','revision','trajectories.mat'),'F','Fshift','t','params');
index=readtable(fullfile(here,'out','revision','continuous-index.csv'));
cascade=readtable(fullfile(here,'out','revision','cascade-index.csv'));
n=height(index);
cols={'H','D','R','Tg','d'};
assert(n==500 && max(abs(index{:,cols}-src.params),[],'all')<1e-12 && ...
    isequal(index{:,cols},cascade{:,cols}));
rng(seed,'twister');
c=randi(3,n,1);
z=randn(n,26);
rows=cell(2*n,15);
for i=1:n
    rows(2*i-1,:)=row(20000+i,"archive","archive_single",c(i),src.params(i,:),src.t,src.F(i,:),z(i,:));
    rows(2*i,:)=row(20000+i,"archive","archive_cascade",c(i),src.params(i,:),src.t,src.Fshift(i,:),z(i,:));
    assert(abs(rows{2*i-1,13}-index.Target_Hz(i))<1e-10 && ...
        abs(rows{2*i,13}-cascade.Target_Hz(i))<1e-10);
end
rows=as_table(rows);
end

function r=row(id,role,scenario,condition,p,t,f,z)
% All three point-prediction inputs and the quality statistic use the same
% 26 observed points. The metadata fields never enter prediction.
vis=t>=1 & t<=1.5;
after=t>1.5;
assert(nnz(vis)==26 && nnz(after)==925 && all(isfinite(f)));
x=f(vis)-50;
scale=sqrt(mean(x.^2));
switch condition
    case 1
        e=zeros(1,26);
    case 2
        e=z*10^(-30/20);
    case {3,5}
        e=z;
        for j=2:26, e(j)=.8*e(j-1)+.6*z(j); end
        if condition==3, e=e*10^(-30/20); else, e=e*.1; end
    case 4
        % At 0.26 s the noise standard deviation triples, then the
        % shape is normalized to a 30 dB full-window expected power.
        w=[ones(1,13),3*ones(1,13)];
        e=z.*(w/sqrt(mean(w.^2)))*10^(-30/20);
end
x=x+scale*e;
u=(t(vis).'-1)/.5;
b=[ones(26,1),u,u.^2]\x.';
r={id,role,scenario,condition,p(1),p(2),p(3),p(4),p(5),x(end),b(2),b(3), ...
    min(f(after)),sqrt(mean(diff(x,2).^2)),sum(after)};
end

function rows=as_table(c)
rows=cell2table(c,VariableNames={'EventID','Role','Scenario','NoiseCode','H','D','R','Tg','d', ...
    'Endpoint','Linear','Curvature','Target_Hz','Quality_Hz','SuffixSamples'});
end
