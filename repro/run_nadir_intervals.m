function run_nadir_intervals()
here = fileparts(mfilename('fullpath'));
project = fullfile(here,'..','Project');
addpath(fullfile(project,'config'),fullfile(project,'core'));
dest = fullfile(here,'out','nadir-intervals');
if ~isfolder(dest), mkdir(dest); end
sys = sys_config();
sim = sim_config();
limits = [2 6;.5 2;.04 .06;.2 1;.02 .08];
sizes = [600 600 1200];
splits = ["scale" "calibration" "test"];
rows = cell(sum(sizes)+2*sizes(end),14);
count = 0;
id = 0;
for split = 1:3
    rng(47100+split,'twister');
    parameters = limits(:,1).' + rand(sizes(split),5).*diff(limits,1,2).';
    condition = randi(3,sizes(split),1);
    noise = randn(sizes(split),26);
    for k = 1:sizes(split)
        id = id+1;
        s = sys; sm = sim;
        p = parameters(k,:);
        s.H=p(1); s.D=p(2); s.R=p(3); s.Tg=p(4); sm.dP=p(5);
        [time,f] = simulate_freq(s,sm);
        cut = sm.t_dist+.5;
        visible = time>=sm.t_dist & time<=cut;
        local = time(visible)-sm.t_dist;
        assert(numel(local)==26);
        [x,target] = observation(time,f,visible,cut,noise(k,:),condition(k));
        count = count+1;
        rows(count,:) = make_row(id,splits(split),"mixed",condition(k),p,local,x,target);
        if split == 3
            [x,target] = observation(time,f,visible,cut,noise(k,:),4);
            count = count+1;
            rows(count,:) = make_row(id,splits(split),"jump",4,p,local,x,target);
            relative = time(time>=sm.t_dist)-sm.t_dist;
            rhs = @(~,z) [(-sm.dP-s.D*z(1)+z(3))/(2*s.H); ...
                (pfr_model(z(1),s)-z(2))/(.25*s.Tg);(z(2)-z(3))/(.75*s.Tg)];
            [~,z] = ode45(rhs,relative,[0;0;0],odeset('RelTol',sm.rel_tol));
            cascade = f;
            cascade(time>=sm.t_dist) = s.f_nom*(1+z(:,1)).';
            [x,target] = observation(time,cascade,visible,cut,noise(k,:),condition(k));
            count = count+1;
            rows(count,:) = make_row(id,splits(split),"cascade",condition(k),p,local,x,target);
        end
    end
end
assert(count==size(rows,1));
output = cell2table(rows,'VariableNames',{'ID','Split','Scenario','Condition','H','D','R','Tg','Deficit', ...
    'Endpoint','Linear','Curvature','Target_Hz','Roughness_Hz'});
writetable(output,fullfile(dest,'scenarios.csv'));
end

function [x,target] = observation(time,f,visible,cut,noise,condition)
x = f(visible)-50;
switch condition
    case 1
        z = zeros(size(noise));
    case 2
        z = noise;
    case 3
        z = noise;
        for k = 2:numel(z), z(k)=.8*z(k-1)+.6*noise(k); end
    otherwise
        w = ones(size(noise)); w(14:end)=3;
        z = noise.*w/sqrt(mean(w.^2));
end
x = x + sqrt(mean(x.^2)/100)*z;
target = min(f(time>cut));
end

function row = make_row(id,split,scenario,condition,p,time,x,target)
u = time(:)/time(end);
c = [ones(size(u)),u,u.^2]\x(:);
roughness = sqrt(mean(diff(x,2).^2));
row = [{id,split,scenario,condition},num2cell(p),{x(end),c(2),c(3),target,roughness}];
end
