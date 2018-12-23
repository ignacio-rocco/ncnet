function parfor_nc4d_PV( qname, dbname, P, RGB, XYZ, params )
dslevel = 8^-1;

[~, dbbasename, ~] = fileparts(dbname);
this_nc4dPV_matname = fullfile(params.output.synth.dir, qname, [dbbasename, params.output.synth.matformat]);

if exist(this_nc4dPV_matname, 'file') ~= 2
    if all(~isnan(P(:)))
        
        %load downsampled images
        Iq = imresize(imread(fullfile(params.data.dir, params.data.q.dir, qname)), dslevel);
        fl = params.data.q.fl * dslevel;
        K = [fl, 0, size(Iq, 2)/2.0; 0, fl, size(Iq, 1)/2.0; 0, 0, 1];
        [ RGBpersp, XYZpersp ] = ht_Points2Persp( RGB, XYZ, K*P, size(Iq, 1), size(Iq, 2) );
        RGB_flag = all(~isnan(XYZpersp), 3);
        
        %compute DSIFT error
        if any(RGB_flag(:))
            %normalization
            Iq_norm = image_normalization( double(rgb2gray(Iq)), RGB_flag );
            I_synth = double(rgb2gray(RGBpersp));
            I_synth(~RGB_flag) = nan;
            I_synth = image_normalization( inpaint_nans(I_synth), RGB_flag );
            
            %compute DSIFT
            [fq, dq] = vl_phow(im2single(Iq_norm),'sizes',8,'step',4);
            [fsynth, dsynth] = vl_phow(im2single(I_synth),'sizes',8,'step',4);
            f_linind = sub2ind(size(I_synth), fsynth(2, :), fsynth(1, :));
            iseval = RGB_flag(f_linind);
            dq = relja_rootsift(single(dq)); dsynth = relja_rootsift(single(dsynth));
            
            %error
            err = sqrt(sum((dq(:, iseval) - dsynth(:, iseval)).^2, 1));
            score = quantile(err, 0.5)^-1;
            errmap = nan(size(I_synth));errmap(f_linind(iseval)) = err;
            xuni = sort(unique(fsynth(1, :)), 'ascend');yuni = sort(unique(fsynth(2, :)), 'ascend');
            errmap = errmap(yuni, xuni);
            
%             %debug
%             figure();set(gcf, 'Position', [0 0 1000, 300]);
%             ultimateSubplot( 3, 1, 1, 1, 0.01, 0.05 );
%             imshow(Iq);
%             ultimateSubplot( 3, 1, 2, 1, 0.01, 0.05 );
%             imshow(RGBpersp);
%             ultimateSubplot( 3, 1, 3, 1, 0.01, 0.05 );
%             imagesc(errmap);colormap('jet');axis image off;
%             keyboard;
            
        else
            score = 0;
            errmap = [];
        end
    else
        Iq = [];
        RGBpersp = [];
        RGB_flag = [];
        score = 0;
        errmap = 0;
    end
    
    if exist(fullfile(params.output.synth.dir, qname), 'dir') ~= 7
        mkdir(fullfile(params.output.synth.dir, qname));
    end
    save(this_nc4dPV_matname, 'Iq', 'RGBpersp', 'RGB_flag', 'score', 'errmap');

    
    I1 = imresize(imread(fullfile(params.data.dir, params.data.q.dir, qname)), dslevel);
    if isempty(RGBpersp)
      I2 = uint8(zeros(size(I1)));
    else
      I2 = RGBpersp;
    end
    
    [I1ylen, I1xlen] = size(I1);
    [I2ylen, I2xlen] = size(I2);

    %cat image
    if I1ylen <= I2ylen
      scale1 = 1;
      scale2 = I1ylen/I2ylen;
      I2 = imresize(I2,scale2);
    else
      scale1 = I2ylen/I1ylen;
      scale2 = 1;
      I1 = imresize(I1,scale1);
    end
    catI = cat(2, I1, I2);
    
    imwrite(catI,[this_nc4dPV_matname '.jpg'],'Quality',90)
    
end


end

