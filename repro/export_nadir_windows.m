function export_nadir_windows()
% Extract the fixed-target short-window inputs from the archived frequency grid.
here = fileparts(mfilename('fullpath'));
archive = load(fullfile(here, 'out', 'revision', 'trajectories.mat'), 'grid');
dest = fullfile(here, 'out', 'nadir-shape');
if ~isfolder(dest), mkdir(dest); end
grid = archive.grid;
first = find(abs(grid.t - 1) < 1e-9);
last = [5 12 25];
edges = {[0 2 4 5], [0 4 8 12], [0 8 16 24]};
blocks = cell(3, 4);
for k = 1:3
    ix = first + (0:last(k));
    t = grid.t(ix);
    x = grid.F(:,ix) - 50;
    u = (t(:)-t(1))/(t(end)-t(1));
    coefficients = [ones(size(u)), u, u.^2] \ x.';
    slopes = zeros(size(x,1), 3);
    fitted = slopes;
    for j = 1:3
        segment = (edges{k}(j):edges{k}(j+1)) + 1;
        centered = t(segment)-mean(t(segment));
        slopes(:,j) = (x(:,segment(end))-x(:,segment(1)))/(t(segment(end))-t(segment(1)));
        fitted(:,j) = (x(:,segment)*centered.')/sum(centered.^2);
    end
    blocks(k,:) = {x(:,end), [x(:,end) coefficients(2:3,:).'], ...
        [x(:,end) slopes], [x(:,end) fitted]};
end
save(fullfile(dest, 'windows-matlab.mat'), 'blocks');
end
