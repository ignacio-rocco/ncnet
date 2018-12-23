
if exist(NC4D_matname, 'file') ~= 2
  %copy results from NC4D
  ImgList = struct('queryname', {}, 'topNname', {}, 'topNscore', {}, 'P', {});
  for ii = 1:Nq

    ImgList(ii).queryname = sorted_list.ImgList(ii).queryname;
    ImgList(ii).topNname = sorted_list.ImgList(ii).topNname(1:pnp_topN);
  
  end
  
  %build list for pnp 
  qlist = cell(1, Nq*pnp_topN);
  dblist = cell(1, Nq*pnp_topN);
  for ii = 1:Nq
    for jj = 1:pnp_topN
      qlist{pnp_topN*(ii-1)+jj} = ImgList(ii).queryname;
      dblist{pnp_topN*(ii-1)+jj} = ImgList(ii).topNname{jj};
    end
  end
  
  %run dense pnp in parpool
  dbfname = fullfile(params.data.dir, params.data.db.cutout.dir, ImgList(ii).topNname{1});
  imsize_db = size(at_imageresize_nc4d(imread(dbfname)));
  parfor ii = 1:Nq
    % load matches for this query
    nc4d_matches = load(fullfile(matches_path,experiment,[num2str(ii) '.mat']),'matches');
    
    for jj = 1:pnp_topN
      %preload query feature
      qfname = fullfile(params.data.dir, params.data.q.dir, ImgList(ii).queryname);
      imsize_q = size(at_imageresize_nc4d(imread(qfname)));
      
      kk = pnp_topN*(ii-1)+jj;
      parfor_NC4D_PE_pnponly( qlist{kk}, dblist{kk}, params, ...
        nc4d_matches.matches(1,jj,:,:), ...
        imsize_q, imsize_db);
      fprintf('nc4dPE: %s vs %s DONE. \n', qlist{kk}, dblist{kk});
    end
  end
  
  %load top pnp
  for ii = 1:Nq
    ImgList(ii).P = cell(1, pnp_topN);
    for jj = 1:pnp_topN
      [~, dbbasename, ~] = fileparts(ImgList(ii).topNname{jj});
      this_nc4dpe_matname = fullfile(params.output.pnp_nc4d_inlier.dir, ImgList(ii).queryname, [dbbasename, params.output.pnp_nc4d.matformat]);
      load(this_nc4dpe_matname, 'P', 'inls');
      ImgList(ii).P{jj} = P;
      ImgList(ii).inls{jj} = inls;
    end
  end
  
  if exist(params.output.dir, 'dir') ~= 7
    mkdir(params.output.dir);
  end
  save('-v6', NC4D_matname, 'ImgList');
else
  load(NC4D_matname, 'ImgList');
end
ImgList_NC4D = ImgList;
