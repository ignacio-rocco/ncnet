%Note: It plots localization rates with varying position thresholds.
set(0,'defaultAxesFontName','times');
set(0,'defaultTextFontName','times');

%% groundtruth
refposes_matname = fullfile(params.gt.dir, params.gt.matname);
load(refposes_matname, 'DUC1_RefList', 'DUC2_RefList');

%% evaluation
vefig = figure();hold on;
linehandles = zeros(1, length(method));
for ii = 1:1:length(method)
    ImgList = method(ii).ImgList;
    
    fp = fopen(sprintf('error_%s.txt',method(ii).description),'w');
    
    %1. DUC1
    poserr_DUC1 = zeros(1, length(DUC1_RefList));
    orierr_DUC1 = zeros(1, length(DUC1_RefList));
    for jj = 1:1:length(DUC1_RefList)
        this_locid = strcmp(DUC1_RefList(jj).queryname, {ImgList.queryname});
        if sum(this_locid) == 0
            poserr_DUC1(jj) = inf;
            orierr_DUC1(jj) = inf;
            keyboard;
        else
            top1_floor = strsplit(ImgList(this_locid).topNname{1}, '/');top1_floor = top1_floor{1};
            isfloorcorrect = strcmp('DUC1', top1_floor);
            if isfloorcorrect && ~isnan(ImgList(this_locid).P{1}(1))
                [poserr_DUC1(jj), orierr_DUC1(jj)] = p2dist(DUC1_RefList(jj).P, ImgList(this_locid).P{1});
            else
                poserr_DUC1(jj) = inf;
                orierr_DUC1(jj) = inf;
            end
            
            fprintf(fp,'%s %f %f\n',DUC1_RefList(jj).queryname,poserr_DUC1(jj),orierr_DUC1(jj));
                        
        end
    end
    
    %2. DUC2
    poserr_DUC2 = zeros(1, length(DUC2_RefList));
    orierr_DUC2 = zeros(1, length(DUC2_RefList));
    for jj = 1:1:length(DUC2_RefList)
        this_locid = strcmp(DUC2_RefList(jj).queryname, {ImgList.queryname});
        if sum(this_locid) == 0
            poserr_DUC2(jj) = inf;
            orierr_DUC2(jj) = inf;
            keyboard;
        else
            top1_floor = strsplit(ImgList(this_locid).topNname{1}, '/');top1_floor = top1_floor{1};
            isfloorcorrect = strcmp('DUC2', top1_floor);
            if isfloorcorrect && ~isnan(ImgList(this_locid).P{1}(1))
                [poserr_DUC2(jj), orierr_DUC2(jj)] = p2dist(DUC2_RefList(jj).P, ImgList(this_locid).P{1});
            else
                poserr_DUC2(jj) = inf;
                orierr_DUC2(jj) = inf;
            end
        end
        
%         if poserr_DUC2(jj) > 1.0
%           fprintf(fp,'%s\n',DUC2_RefList(jj).queryname);
%         end
        
        fprintf(fp,'%s %f %f\n',DUC2_RefList(jj).queryname,poserr_DUC2(jj),orierr_DUC2(jj));
        
    end
    fclose(fp);
    
    %3. all
    poserr_all = [poserr_DUC1, poserr_DUC2];
    orierr_all = [orierr_DUC1, orierr_DUC2];
    

    %plot curve
    eval_poserr = poserr_all;
    eval_orierr = orierr_all*180/pi;
    max_orierr = 10;%deg
    eval_poserr(eval_orierr>max_orierr) = inf;
    eval_err = sort(eval_poserr, 'ascend');
    errthr_dot = [0:0.0625:1, 1.125:0.125:2];
    localized_rate_dot = sum(repmat(eval_err', 1, size(errthr_dot, 2)) < repmat(errthr_dot, size(eval_err, 2), 1), 1) / size(eval_err, 2);
    
    figure(vefig);
    linehandles(ii) = plot(errthr_dot, localized_rate_dot*100, method(ii).marker, 'LineWidth', 2.0);
    set(linehandles(ii), 'MarkerFaceColor', get(linehandles(ii), 'Color'));
    drawnow;
    
    
end

figure(vefig);
xlim([0 2]);ylim([0 80]);grid on;
% xlim([0 0.5]);ylim([0 80]);grid on;
lgnd = legend(linehandles, {method.description}, 'Location', 'southeast', 'FontSize', 10, 'Interpreter', 'latex');
lgnd.FontName = 'Times New Roman';
xlabel('Distance threshold [meters]');ylabel('Correctly localized queries [%]');
set(gca, 'FontSize', 18);
% set(gca, 'XTick', 0:0.5:3);
set(gca, 'XTick', 0:0.25:2);
% set(gca, 'XTick', 0:0.1:0.5);
set(get(gcf,'CurrentAxes'),'Position',[0.15 0.13 0.8 0.8]);
set(gcf,'PaperUnits','Inches','PaperPosition',[0 0 5 5]);
drawnow;

%save fig
figname = fullfile(params.output.dir, sprintf('athr%.4f_%d.fig', max_orierr, length(poserr_all)));
epsname = fullfile(params.output.dir, sprintf('athr%.4f_%d.eps', max_orierr, length(poserr_all)));

print(vefig, '-depsc',epsname,'-r160');
savefig(vefig, figname);

