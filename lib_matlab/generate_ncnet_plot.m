
ncnet=load(NC4D_matname);

if do_densePV
	ncnet_dpv=load(nc4dPV_matname);
end

% plot
method = struct();
i=1
method(i).ImgList = ncnet.ImgList;
method(i).description = 'DensePE + NCNet';
method(i).marker = '--b';
if do_densePV
	i=i+1
	method(i).ImgList = ncnet_dpv.ImgList;
	method(i).description = 'InLoc + NCNet';
	method(i).marker = '--c';
end

ht_plotcurve_WUSTL
