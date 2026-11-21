function nadir_ecgr(stage, out)
% MATLAB feature preparation: archived development or locked confirmation.
here = fileparts(mfilename('fullpath'));
c = jsondecode(fileread(fullfile(out,'method-protocol.json')));
q = c.confirmation;
src = fullfile(here,'out','revision');
map = operators(c);
switch string(stage)
    case "develop"
        unit_checks(map,out);
        A = load(fullfile(src,'trajectories.mat'),'grid');
        grid = readtable(fullfile(src,'grid-index.csv'),TextType='string');
        t = A.grid.t;
        vis = find(t>=q.disturbance_s & t<=q.disturbance_s+q.cutoff_relative_s);
        assert(numel(vis)==26);
        clean = A.grid.F(:,vis);
        training = grid.Split=="train";
        validation = find(grid.Split=="validation");
        [~, beta] = estimate(clean(training,:),map,0,1);
        % Statistical shrinkage target calibrated only on original training inputs.
        prior = max(mean(beta(:,2).^2),c.prior_second_moment_floor_hz2);
        total = [clean;repmat(clean(validation,:),c.validation_noise_repeats,1)];
        id = [grid.ID;repmat(grid.ID(validation),c.validation_noise_repeats,1)];
        rep = [zeros(height(grid),1);repelem((1:c.validation_noise_repeats).',numel(validation))];
        rs = RandStream('mt19937ar','Seed',c.validation_noise_seed);
        rows = rep>0;
        noisy = total(rows,:);
        sig = sqrt(mean((noisy-50).^2,2))*10^(-q.white_snr_db/20);
        total(rows,:) = noisy+sig.*randn(rs,size(noisy));
        X = base_features(total);
        for k = 1:numel(c.lambda_candidates)
            X = [X,estimate(total,map,c.lambda_candidates(k),prior)]; %#ok<AGROW>
        end
        writematrix([id rep X],fullfile(out,'development-features.csv'));
        writematrix(prior,fullfile(out,'training-cubic-moment.csv'));
    case "confirm"
        lock = jsondecode(fileread(fullfile(out,'freeze.json')));
        A = load(fullfile(out,'trajectories.mat'),'traceA','traceB','vis');
        input = readtable(fullfile(out,'features.csv'),TextType='string');
        n = height(input);
        raw = zeros(n,26);
        % Replay the exact same member windows and white-noise draws as the baseline export.
        for i = 1:n
            ref = input.ReferenceID(i);
            id = input.PairID(i);
            if input.Member(i)==1
                f = A.traceA(ref,A.vis);
                seed = q.noise_seed+ref;
            else
                f = A.traceB(id,A.vis);
                seed = q.noise_seed+size(A.traceA,1)+id;
            end
            if input.Rep(i)>0
                rs = RandStream('mt19937ar','Seed',seed);
                z = randn(rs,q.noise_repeats,26);
                f = f+sqrt(mean((f-50).^2))*10^(-q.white_snr_db/20)*z(input.Rep(i),:);
            end
            raw(i,:) = f;
        end
        base = base_features(raw);
        saved = input{:,{'Endpoint','InitialRate','Linear','Curvature','Slope1','Slope2','Slope3'}};
        assert(max(abs(base-saved),[],'all')<1e-10,'Confirmation windows differ from baseline.');
        x = estimate(raw,map,lock.lambda,lock.training_cubic_second_moment_hz2);
        T = input(:,{'ReferenceID','PairID','Member','Condition','Rep'});
        T = [T,array2table(x,VariableNames={'Endpoint','Symmetric','Antisymmetric'})];
        writetable(T,fullfile(out,'ecgr-features.csv'));
    otherwise
        error('Unknown ECGR stage.');
end
end

function op = operators(c)
% L maps a 26-point window to its 24 interior endpoint-chord residuals.
u = linspace(0,1,26).';
v = u(2:end-1);
L = [-(1-v),eye(numel(v)),-v];
B = [v.*(1-v),v.*(1-v).*(2*v-1)];
S = c.working_ar1_weight*toeplitz(c.working_ar1_rho.^(0:25))+(1-c.working_ar1_weight)*eye(26);
C = chol(L*S*L','lower');
op = struct('L',L,'B',B,'C',C,'W',C\L,'Z',C\B,'S',S);
op.gram = op.Z'*op.Z;
op.LS = op.gram\(op.Z'*op.W);
op.rough = op.W-op.Z*op.LS;
end

function [x,b,proxy] = estimate(f,op,lambda,prior)
% Beta=(G+diag(0,lambda*proxy/prior))\z is a two-variable
% solve after removing the endpoint chord. The roughness proxy depends only
% on the unmodeled component of the same causal window, in Hz squared.
b0 = f*op.LS.';
proxy = sum((f*op.rough.').^2,2)/(size(op.Z,1)-2);
g = op.gram;
z = b0*g;
p = lambda*proxy/prior;
determinant = g(1,1)*(g(2,2)+p)-g(1,2)^2;
b = [(z(:,1).*(g(2,2)+p)-z(:,2)*g(1,2))./determinant, ...
     (z(:,2)*g(1,1)-z(:,1)*g(1,2))./determinant];
x = [f(:,end)-50,b];
end

function x = base_features(f)
t = (0:25)*.02;
u = t(:)/t(end);
z = f-50;
beta = [ones(26,1),u,u.^2]\z.';
edges = [1 9 17 25];
slopes = zeros(size(f,1),3);
for k=1:3
    i = edges(k):edges(k+1);
    tc = t(i)-mean(t(i));
    slopes(:,k) = z(:,i)*tc.'/(tc*tc.');
end
x = [z(:,26),(z(:,6)-z(:,1))/.10,beta(2:3,:).',slopes];
end

function unit_checks(op,out)
u = linspace(0,1,26);
shape = [zeros(1,2);op.B;zeros(1,2)];
assert(all(shape([1,end],:)==0,'all'));
assert(norm(op.L*ones(26,1))<1e-12 && norm(op.L*u.')<1e-12);
[x,b] = estimate([50*ones(1,26);50-.15*u],op,1,.01);
assert(max(abs(b),[],'all')<1e-11 && abs(x(2,1)+.15)<1e-12);
clean = 50-.15*u+[.14 .06]*shape.';
[~,b] = estimate(clean,op,1,.01);
assert(max(abs(b-[.14 .06]))<1e-11);
% Construct a window perturbation that leaves baseline OLS coefficients and
% both endpoints unchanged. Higher-order roughness should change ECGR.
Q = [ones(26,1),u.',u.'.^2];
constraints = [pinv(Q); eye(1,26); fliplr(eye(1,26))];
P = null(constraints);
j = P*(P'*(cos(18*pi*u.')));
j = .04*j/norm(j,inf);
[x1,b1] = estimate(clean,op,4,.001);
[x2,b2] = estimate(clean+j.',op,4,.001);
first = base_features(clean); second = base_features(clean+j.');
assert(max(abs(first([1 3 4])-second([1 3 4])))<1e-10);
assert(abs(b1(2)-b2(2))>1e-5 && x1(1)==x2(1));
% Separate synthetic white and AR(1) checks: verify propagated covariance
% against Monte Carlo and that the mean unregularized signal is unbiased.
rs = RandStream('mt19937ar','Seed',1);
N = 14000;
Z = randn(rs,N,26);
S = {eye(26),toeplitz(.65.^(0:25))};
for k=1:2
    z = Z*chol(S{k});
    empirical = cov(z*op.L.');
    theory = op.L*S{k}*op.L.';
    assert(norm(empirical-theory,'fro')/norm(theory,'fro')<.06);
    [~,b] = estimate(clean+.01*z,op,0,.001);
    assert(max(abs(mean(b)-[.14 .06]))<.002);
    [~,bs,p] = estimate(clean+.01*z,op,4,.001);
    assert(median(abs(bs(:,2)))<median(abs(b(:,2))) && all(p>0));
end
writelines(["PASS affine, zero residual, endpoints and exact cubic shape"; ...
    "PASS quadratic baseline invariance under an ECGR-changing perturbation"; ...
    "PASS propagated white / AR(1) covariance, unshrunk bias check and adaptive shrinkage"], ...
    fullfile(out,'matlab-unit-checks.log'));
end
