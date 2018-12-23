function parfor_NC4D_PE_pnponly( qname, dbname, params, matches, imsize_q, imsize_db)

[~, dbbasename, ~] = fileparts(dbname);
this_nc4dpe_matname = fullfile(params.output.pnp_nc4d_inlier.dir, qname, [dbbasename, params.output.pnp_nc4d.matformat]);

if exist(this_nc4dpe_matname, 'file') ~= 2
    
  %geometric verification results
  
  M = squeeze(matches);
  
  f1 = M(:,1:2)';
  f2 = M(:,3:4)';
  scr = M(:,5);
  
  thr = params.ncnet.thr;
  f1 = f1(:,scr>thr);
  f2 = f2(:,scr>thr);
  
  if isfield(params.ncnet,'N_subsample')
      subsample=randperm(size(f1,2));
      subsample=subsample(1:min(size(f1,2),params.ncnet.N_subsample));
      f1=f1(:,subsample);
      f2=f2(:,subsample);
  end
  
  tent_xq2d = f1;
  tent_xdb2d = f2;
 
  %depth information
  this_db_matname = fullfile(params.data.dir, params.data.db.cutout.dir, [dbname, params.data.db.cutout.matformat]);
  load(this_db_matname, 'XYZcut');
  %load transformation matrix (local to global)
  this_floorid = strsplit(dbname, '/');this_floorid = this_floorid{1};
  info = parse_WUSTL_cutoutname( dbname );
  transformation_txtname = fullfile(params.data.dir, params.data.db.trans.dir, this_floorid, 'transformations', ...
    sprintf('%s_trans_%s.txt', info.scene_id, info.scan_id));
  [ ~, P_after ] = load_WUSTL_transformation(transformation_txtname);
  
  %Feature upsampling
  Iqsize = size(imread(fullfile(params.data.dir, params.data.q.dir, qname)));
  Idbsize = size(XYZcut);
  
  tent_xq2d(1,:) = Iqsize(2)*tent_xq2d(1,:); 
  tent_xq2d(2,:) = Iqsize(1)*tent_xq2d(2,:);
  tent_xdb2d(1,:) = floor(Idbsize(2)*tent_xdb2d(1,:));
  tent_xdb2d(2,:) = floor(Idbsize(1)*tent_xdb2d(2,:));
  tent_xdb2d(1,tent_xdb2d(1,:) == 0) = 1; % fix zeros
  tent_xdb2d(2,tent_xdb2d(2,:) == 0) = 1;
  
 
  %query ray
  Kq = [params.data.q.fl, 0, Iqsize(2)/2.0; ...
    0, params.data.q.fl, Iqsize(1)/2.0; ...
    0, 0, 1];
  tent_ray2d = Kq^-1 * [tent_xq2d; ones(1, size(tent_xq2d, 2))];
  %DB 3d points
  indx = sub2ind(size(XYZcut(:,:,1)),tent_xdb2d(2,:),tent_xdb2d(1,:));
  X = XYZcut(:,:,1);Y = XYZcut(:,:,2);Z = XYZcut(:,:,3);
  tent_xdb3d = [X(indx); Y(indx); Z(indx)];
  tent_xdb3d = bsxfun(@plus, P_after(1:3, 1:3)*tent_xdb3d, P_after(1:3, 4));
  %Select keypoint correspond to 3D
  idx_3d = all(~isnan(tent_xdb3d), 1);
  tent_xq2d = tent_xq2d(:, idx_3d);
  tent_xdb2d = tent_xdb2d(:, idx_3d);
  tent_ray2d = tent_ray2d(:, idx_3d);
  tent_xdb3d = tent_xdb3d(:, idx_3d);
    
  tentatives_2d = [tent_xq2d; tent_xdb2d];
  tentatives_3d = [tent_ray2d; tent_xdb3d];  
  
  %solver
  if size(tentatives_2d, 2) < 3
    P = nan(3, 4);
    inls = false(1, size(tentatives_2d, 2));
  else
    [ P, inls ] = ht_lo_ransac_p3p( tent_ray2d, tent_xdb3d, params.ncnet.pnp_thr*pi/180, 10000);
    if isempty(P)
      P = nan(3, 4);
    end
  end
  
  if exist(fullfile(params.output.pnp_nc4d_inlier.dir, qname), 'dir') ~= 7
    mkdir(fullfile(params.output.pnp_nc4d_inlier.dir, qname));
  end
  save('-v7.3', this_nc4dpe_matname, 'P', 'inls', 'tentatives_2d', 'tentatives_3d', 'idx_3d');
  
  if 0
    %     %debug
    close all;
    
    Iq = imread(fullfile(params.data.dir, params.data.q.dir, qname));
    Idb = imread(fullfile(params.data.dir, params.data.db.cutout.dir, dbname));
    points.x2 = tentatives_2d(3, :);
    points.y2 = tentatives_2d(4, :);
    points.x1 = tentatives_2d(1, :);
    points.y1 = tentatives_2d(2, :);
    points.color = 'g';
    points.facecolor = 'g';
    points.markersize = 10;
    points.linestyle = '-';
    points.linewidth = 0.5;
    show_matches2_horizontal( Iq, Idb, points, ...
      [1:size(tentatives_2d,2); 1:size(tentatives_2d,2)], inls );

    if ~exist('eg','dir'), mkdir('eg'); end
    print('-dpng',['eg/' qname '.png'],'-r60');
%     keyboard;
  end
end


end

