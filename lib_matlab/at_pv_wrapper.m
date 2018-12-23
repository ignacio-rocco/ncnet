function at_pv_wrapper(ii,dbscanlist_uniq,dbscantranslist_uniq,qlist_uniq,dblist_uniq,Plist_uniq,params)

this_dbscan = dbscanlist_uniq{ii};
this_dbscantrans = dbscantranslist_uniq{ii};
this_qlist = qlist_uniq{ii};
this_dblist = dblist_uniq{ii};
this_Plist = Plist_uniq{ii};
%load scan
load(fullfile(params.data.dir, params.data.db.scan.dir, this_dbscan), 'A');

[ ~, P_after ] = load_WUSTL_transformation(fullfile(params.data.dir, params.data.db.trans.dir, this_dbscantrans));
RGB = [A{5}, A{6}, A{7}]';
XYZ = [A{1}, A{2}, A{3}]';
XYZ = P_after * [XYZ; ones(1, length(XYZ))];
XYZ = bsxfun(@rdivide, XYZ(1:3, :), XYZ(4, :));

%compute synthesized images and similarity scores
for jj = 1:1:length(this_qlist)
  parfor_nc4d_PV( this_qlist{jj}, this_dblist{jj}, this_Plist{jj}, RGB, XYZ, params );
  %             fprintf('densePV: %d / %d done. \n', jj, length(this_qlist));
end
fprintf('ncnetPV: scan %s (%d / %d) done. \n', this_dbscan, ii, length(dbscanlist_uniq));
