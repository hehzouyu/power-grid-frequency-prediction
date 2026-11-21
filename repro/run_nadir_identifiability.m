function run_nadir_identifiability()
here = fileparts(mfilename('fullpath'));
dest = fullfile(here,'out','nadir-identifiability');
if ~isfolder(dest), mkdir(dest); end
source = readtable(fullfile(here,'out','revision','continuous-index.csv'));
anchors = source(round(linspace(1,height(source),60)),:);
p = anchors{:,{'H','D','R','Tg','d'}};
t = 0:.02:19;
observed = zeros(height(anchors),26);
jacobian = zeros(height(anchors),26,5,2);
target_jacobian = zeros(height(anchors),5,2);
limited = false(height(anchors),1);
anchor_target = zeros(height(anchors),1);
scaling_error = nan(height(anchors),1);
limits = [2 6;.5 2;.04 .06;.2 1;.02 .08];
steps = [.001 .0005];
for i = 1:height(anchors)
    [f,limited(i)] = response(t,p(i,:));
    observed(i,:) = f(1:26);
    anchor_target(i) = min(f(27:end));
    for s = 1:numel(steps)
        h = steps(s);
        for j = 1:5
            plus = p(i,:); minus = plus;
            plus(j) = plus(j)*exp(h); minus(j) = minus(j)*exp(-h);
            [high,phit] = response(t,plus);
            [low,mhit] = response(t,minus);
            limited(i) = limited(i) || phit || mhit;
            jacobian(i,:,j,s) = (high(1:26)-low(1:26))/(2*h);
            target_jacobian(i,j,s) = (min(high(27:end))-min(low(27:end)))/(2*h);
        end
    end
    transformed = p(i,:).*[1.05 1.05 1/1.05 1 1.05];
    if ~limited(i) && all(transformed>=limits(:,1).') && all(transformed<=limits(:,2).')
        [g,hit] = response(t,transformed);
        if ~hit, scaling_error(i)=max(abs(f-g)); end
    end
end
ids = anchors.ID;
save(fullfile(dest,'sensitivities.mat'),'ids','p','steps','observed','limited','anchor_target', ...
    'jacobian','target_jacobian','scaling_error');

settings = {[2.5 3 5 5.5],[.6 .8 1.5 1.9],[.25 .35 .7 1]};
coordinates = [1 2 4];
names = ["H" "D" "Tg"];
rows = cell(36,8);
n = 0;
for d = [.025 .035 .05]
    ref = [4 1 .05 .5 d];
    [base,basehit] = response(t,ref);
    for k = 1:3
        for value = settings{k}
            n = n+1;
            modified = ref;
            modified(coordinates(k))=value;
            func = @(deficit) endpoint_error(modified,deficit,base(26));
            if func(.02)*func(.08)>0
                rows(n,:) = {d,names(k),value,NaN,NaN,NaN,NaN,NaN};
                continue
            end
            adjusted = fzero(func,[.02 .08],optimset('TolX',1e-10));
            modified(5)=adjusted;
            [other,hit] = response(t,modified);
            rows(n,:) = {d,names(k),value,adjusted,other(26)-base(26), ...
                min(other(27:end))-min(base(27:end)),sqrt(mean((other(1:26)-base(1:26)).^2)),double(basehit||hit)};
        end
    end
end
output = cell2table(rows,'VariableNames',{'ReferenceDeficit','Factor','FactorValue','MatchedDeficit', ...
    'EndpointError_Hz','TargetDifference_Hz','ShapeRMS_Hz','EitherLimited'});
writetable(output,fullfile(dest,'matched-scan.csv'));
end

function error = endpoint_error(p,d,target)
p(5)=d;
f = response(0:.02:.5,p);
error = f(end)-target;
end

function [f,limited] = response(t,p)
M=2*p(1); D=p(2); R=p(3); Tg=p(4); d=p(5);
rhs = @(~,z)[(-d-D*z(1)+z(2))/M;(min(max(-z(1)/R,0),.1)-z(2))/Tg];
[~,z] = ode45(rhs,t,[0;0],odeset('RelTol',1e-10,'AbsTol',1e-12));
f = 50*(1+z(:,1)).';
limited = any(-z(:,1)/R>=.1);
end
