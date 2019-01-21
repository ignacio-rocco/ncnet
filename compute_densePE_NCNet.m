% Evaluate NCNet matches on top of densePE shortlist

% adjust path and experiment name
inloc_demo_path = '/sequoia/data1/iroccosp/GIT/github/InLoc_demo_old/';
experiment = 'densePE_top100_shortlist_cvpr18_SZ_NEW_3200_K_2_BOTHDIRS_SOFTMAX_CHECKPOINT_ncnet_ivd';

if exist('ncnet_path')==0
	ncnet_path=pwd;
end
matches_path = fullfile(ncnet_path,'matches');

sorted_list_fn = 'densePE_top100_shortlist_cvpr18.mat';
sorted_list = load(fullfile(ncnet_path,'datasets/inloc/',sorted_list_fn));

addpath(fullfile(ncnet_path,'lib_matlab'));

% init paths
cd(inloc_demo_path)
startup;
[ params ] = setup_project_ht_WUSTL;
% add extra parameters
params.output.dir = ['outputs_' experiment '_' sorted_list_fn(1:end-4)];
params.output.gv_nc4d.dir = fullfile(params.output.dir, 'gv_nc4d'); % dense matching results path
params.output.gv_nc4d.matformat = '.gv_nc4d.mat'; % dense matching results 
params.output.pnp_nc4d.matformat = '.pnp_nc4d_inlier.mat'; % PnP results 
% redefine gt poses path
params.gt.dir = fullfile(ncnet_path,'lib_matlab')

Nq=length(sorted_list.ImgList);

pnp_topN=10;
% set parameters
params.ncnet.thr = 0.75;
params.ncnet.pnp_thr = 0.2;
params.output.pnp_nc4d_inlier.dir = fullfile(params.output.dir, ...
  sprintf('top_%i_PnP_thr%03d_rthr%03d',pnp_topN,params.ncnet.thr*100,params.ncnet.pnp_thr*100));
NC4D_matname = fullfile(params.output.dir, ...
  sprintf('top_%i_thr%03d_rthr%03d.mat',...
  pnp_topN,params.ncnet.thr*100,params.ncnet.pnp_thr*100));

% compute poses from matches
ir_top100_NC4D_localization_pnponly;

do_densePV=true

if do_densePV
  params.output.synth.dir = fullfile(params.output.dir, ...
  sprintf('top_%i_thr%03d_rthr%03d_densePV'));

  nc4dPV_matname = fullfile(params.output.dir, ...
  sprintf('top_%i_thr%03d_rthr%03d_densePV.mat',...
  pnp_topN,params.ncnet.thr*100,params.ncnet.pnp_thr*100));
  % run pose verification by rendering sythetic views
  ht_top10_NC4D_PV_localization
end

generate_ncnet_plot