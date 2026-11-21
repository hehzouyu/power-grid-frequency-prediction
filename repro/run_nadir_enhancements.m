function run_nadir_enhancements()
here = fileparts(mfilename('fullpath'));
src = fullfile(here,'out','revision');
dest = fullfile(here,'out','nadir-enhancements');
if ~isfolder(dest), mkdir(dest); end
a = load(fullfile(src,'trajectories.mat'));
index = readtable(fullfile(src,'grid-index.csv'));
ids = find(strcmp(index.Split,'test'));
assert(numel(ids)==232);
win = a.grid.t>=1 & a.grid.t<=1.5;
t = a.grid.t(win)-1;
f = a.grid.F(ids,win);
power = mean((f-50).^2,2)/100;
rng(61320,'twister');
blocks = cell(10,3,4);
variance = zeros(10,3);
weights = ones(size(t));
weights(t>.25) = 3;
weights = weights/sqrt(mean(weights.^2));
for rep = 1:10
    white = randn(size(f));
    colored = white;
    for k = 2:size(f,2)
        colored(:,k) = .8*colored(:,k-1)+.6*white(:,k);
    end
    deviations = {white,colored,white.*weights};
    for j = 1:3
        variance(rep,j) = mean(deviations{j}.^2,'all');
        noisy = f+sqrt(power).*deviations{j};
        for k = 1:4
            blocks{rep,j,k} = extract_feature(t,noisy,k);
        end
    end
end
save(fullfile(dest,'noise-features.mat'),'blocks','ids','variance');
ms = zeros(200,4);
for k = 1:4
    extract_feature(t,f(1,:),k);
    for rep = 1:200
        clock = tic;
        extract_feature(t,f(1,:),k);
        ms(rep,k) = 1e3*toc(clock);
    end
end
writematrix(ms,fullfile(dest,'feature-timing-ms.csv'));
writelines([string(version); string(computer)],fullfile(dest,'matlab-environment.txt'));

time = 0:.02:19;
scan = [];
for d = [.025 .035 .050]
    base = simulate(time,[4 1 .05 .2 d],false);
    endpoint = base.Frequency_Hz(26);
    future = min(base.Frequency_Hz(27:end));
    for tg = [.35 .50 .70 1.00]
        gap = @(deficit) mismatch([4 1 .05 tg deficit],endpoint);
        if gap(.02)*gap(.08) > 0
            scan(end+1,:) = [d tg NaN NaN NaN NaN NaN]; %#ok<AGROW>
            continue
        end
        adjusted = fzero(gap,[.02 .08],optimset('TolX',1e-10));
        other = simulate(time,[4 1 .05 tg adjusted],false);
        scan(end+1,:) = [d tg adjusted other.Frequency_Hz(26)-endpoint ...
            future min(other.Frequency_Hz(27:end)) any(base.Limited|other.Limited)]; %#ok<AGROW>
    end
end
writetable(array2table(scan,'VariableNames',{'BaseDeficit_pu','Tg_s','MatchedDeficit_pu', ...
    'EndpointError_Hz','BaseTarget_Hz','MatchedTarget_Hz','EitherLimited'}),fullfile(dest,'endpoint-scan.csv'));

pairs = readtable(fullfile(here,'out','nadir-shape','paired-structure.csv'));
pairs = sortrows(pairs(strcmp(pairs.Feature,'fitted_segments'),:),{'AbsoluteError','ID'});
selected = pairs(round(height(pairs)*[.10 .50 .95]),:);
state = table();
for k = 1:height(selected)
    row = selected(k,:);
    params = row{1,{'H','D','R','Tg','d'}};
    for cascade = [false true]
        result = simulate(time,params,cascade);
        original = a.F;
        if cascade, original = a.Fshift; end
        replay = max(abs(result.Frequency_Hz-original(row.ID,a.t>=1).'));
        assert(replay < 5e-5,'Replay difference exceeds 0.05 mHz.');
        result.MaxReplayError_Hz = repmat(replay,height(result),1);
        result.ID(:) = row.ID;
        result.Cascade(:) = cascade;
        state = [state;result]; %#ok<AGROW>
    end
end
writetable(selected,fullfile(dest,'structure-selected.csv'));
writetable(state,fullfile(dest,'structure-states.csv'));
end

function output = extract_feature(t,f,kind)
x = f-50;
endpoint = x(:,end);
switch kind
    case 1
        output = endpoint;
    case 2
        u = t(:)/t(end);
        c = [ones(size(u)),u,u.^2]\x.';
        output = [endpoint,c(2:3,:).'];
    otherwise
        edges = [1 9 17 25];
        slopes = zeros(size(x,1),3);
        for j = 1:3
            k = edges(j):edges(j+1);
            if kind == 3
                slopes(:,j) = (x(:,k(end))-x(:,k(1)))/(t(k(end))-t(k(1)));
            else
                centered = t(k)-mean(t(k));
                slopes(:,j) = x(:,k)*centered.'/sum(centered.^2);
            end
        end
        output = [endpoint,slopes];
end
end

function delta = mismatch(params,endpoint)
result = simulate(0:.02:.5,params,false);
delta = result.Frequency_Hz(end)-endpoint;
end

function result = simulate(t,params,cascade)
M = 2*params(1); D = params(2); R = params(3); Tg = params(4); d = params(5);
command = @(x) min(max(-x/R,0),.1);
if cascade
    rhs = @(~,x) [(-d-D*x(1)+x(3))/M;(command(x(1))-x(2))/(.25*Tg);(x(2)-x(3))/(.75*Tg)];
    init = [0;0;0];
else
    rhs = @(~,x) [(-d-D*x(1)+x(2))/M;(command(x(1))-x(2))/Tg];
    init = [0;0];
end
[time,y] = ode45(rhs,t,init,odeset('RelTol',1e-9,'AbsTol',1e-11));
result = table(time,50*(1+y(:,1)),command(y(:,1)),y(:,end),-y(:,1)/R>=.1,zeros(size(time)), ...
    false(size(time)),'VariableNames',{'Time_s','Frequency_Hz','Command_pu','GovernorPower_pu','Limited','ID','Cascade'});
end
