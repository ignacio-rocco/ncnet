%Note: It first synthesize query views according to top10 pose candedates
%and compute similarity between original query and synthesized views. Pose
%candidates are then re-scored by the similarity.

%% densePV (top10 pose candidate -> pose verification)
PV_topN = 10;
% nc4dPV_matname = fullfile(params.output.dir, 'densePV_top10_shortlist.mat');
if exist(nc4dPV_matname, 'file') ~= 2
    
    %synthesis list
    qlist = cell(1, PV_topN*length(ImgList_NC4D));
    dblist = cell(1, PV_topN*length(ImgList_NC4D));
    Plist = cell(1, PV_topN*length(ImgList_NC4D));
    for ii = 1:1:length(ImgList_NC4D)
        for jj = 1:1:PV_topN
            qlist{PV_topN*(ii-1)+jj} = ImgList_NC4D(ii).queryname;
            dblist{PV_topN*(ii-1)+jj} = ImgList_NC4D(ii).topNname{jj};
            Plist{PV_topN*(ii-1)+jj} = ImgList_NC4D(ii).P{jj};
        end
    end
    %find unique scans
    dbscanlist = cell(size(dblist));
    dbscantranslist = cell(size(dblist));
    for ii = 1:1:length(dblist)
        this_floorid = strsplit(dblist{ii}, '/');this_floorid = this_floorid{1};
        info = parse_WUSTL_cutoutname( dblist{ii} );
        dbscanlist{ii} = fullfile(this_floorid, [info.scene_id, '_scan_', info.scan_id, params.data.db.scan.matformat]);
        dbscantranslist{ii} = fullfile(this_floorid, 'transformations', [info.scene_id, '_trans_', info.scan_id, '.txt']);
    end
    [dbscanlist_uniq, sort_idx, uniq_idx] = unique(dbscanlist);
    dbscantranslist_uniq = dbscantranslist(sort_idx);
    qlist_uniq = cell(size(dbscanlist_uniq));
    dblist_uniq = cell(size(dbscanlist_uniq));
    Plist_uniq = cell(size(dbscanlist_uniq));
    for ii = 1:1:length(dbscanlist_uniq)
        idx = uniq_idx == ii;
        qlist_uniq{ii} = qlist(idx);
        dblist_uniq{ii} = dblist(idx);
        Plist_uniq{ii} = Plist(idx);
    end
    
    %compute synthesized views and similarity
    parfor ii = 1:1:length(dbscanlist_uniq)
      at_pv_wrapper(ii,dbscanlist_uniq,dbscantranslist_uniq,qlist_uniq,dblist_uniq,Plist_uniq,params)
    end
    
    %load similarity score and reranking
    ImgList = struct('queryname', {}, 'topNname', {}, 'topNscore', {}, 'P', {});
    for ii = 1:1:length(ImgList_NC4D)
        ImgList(ii).queryname = ImgList_NC4D(ii).queryname;
        ImgList(ii).topNname = ImgList_NC4D(ii).topNname(1:PV_topN);
        ImgList(ii).topNscore = zeros(1, PV_topN);
        ImgList(ii).P = ImgList_NC4D(ii).P(1:PV_topN);
        for jj = 1:1:PV_topN
            [~, dbbasename, ~] = fileparts(ImgList(ii).topNname{jj});
            load(fullfile(params.output.synth.dir, ImgList(ii).queryname, [dbbasename, params.output.synth.matformat]), 'score');
            ImgList(ii).topNscore(jj) = score;
        end
        
        %reranking
        [sorted_score, idx] = sort(ImgList(ii).topNscore, 'descend');
        ImgList(ii).topNname = ImgList(ii).topNname(idx);
        ImgList(ii).topNscore = ImgList(ii).topNscore(idx);
        ImgList(ii).P = ImgList(ii).P(idx);
    end
    
    if exist(params.output.dir, 'dir') ~= 7
        mkdir(params.output.dir);
    end
    save('-v6', nc4dPV_matname, 'ImgList');
    
else
    load(nc4dPV_matname, 'ImgList');
end
ImgList_NC4DPV = ImgList;
