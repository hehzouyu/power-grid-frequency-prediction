function save_results(fig_handle, fig_name, tbl, tbl_name, out_dir)
% SAVE_RESULTS  统一保存图片和表格到 results/ 目录
% 大白话:每次实验的图和表都手动另存为太麻烦,而且容易把结果散落各处。
%        这个函数规定:图统一进 figures/,表统一进 tables/,命名用时间戳防覆盖。
%
% 输入:
%   fig_handle : 图窗句柄(用 gcf 传入当前图)
%   fig_name   : 图片文件名(不含扩展名)
%   tbl        : 要保存的表格(可传 [] 跳过)
%   tbl_name   : 表格文件名(不含扩展名)
%   out_dir    : results 目录路径

if ~exist(fullfile(out_dir, 'figures'), 'dir')
    mkdir(fullfile(out_dir, 'figures'));
end
if ~exist(fullfile(out_dir, 'tables'), 'dir')
    mkdir(fullfile(out_dir, 'tables'));
end

% 图片:PNG 用于查看,.fig 用于后续在MATLAB里继续编辑
exportgraphics(fig_handle, fullfile(out_dir, 'figures', [fig_name '.png']), 'Resolution', 150);
savefig(fig_handle, fullfile(out_dir, 'figures', [fig_name '.fig']));

% 表格
if ~isempty(tbl)
    writetable(tbl, fullfile(out_dir, 'tables', [tbl_name '.csv']));
end
end
