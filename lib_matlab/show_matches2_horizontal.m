function [ h ] = show_matches2_horizontal( I1, I2, showkeys, match12, inls12 )
%

if size(I1, 3) == 3
  I1 = rgb2gray(I1);
end
if size(I2, 3) == 3
  I2 = rgb2gray(I2);
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

h = figure('Visible','off');
imagesc(catI);
colormap(gray);
set(gca, 'Position', [0 0 1 1]);
set(gcf, 'Position', [0 0 size(catI,2) size(catI,1)]);
grid off;
axis equal tight;
axis off;
hold on;

%plot matches
style = length(showkeys);
for s = 1:1:style
  x1 = scale1*showkeys(s).x1(1,match12(1,:));
  y1 = scale1*showkeys(s).y1(1,match12(1,:));
  x2 = scale2*showkeys(s).x2(1,match12(2,:)) + size(I1,2) + 10;
  y2 = scale2*showkeys(s).y2(1,match12(2,:));
  
  mh = scatter([x1'; x2'], [y1'; y2']);
  set(mh, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', 'SizeData', 10);
  
  x1 = scale1*showkeys(s).x1(1,inls12(1,:));
  y1 = scale1*showkeys(s).y1(1,inls12(1,:));
  x2 = scale2*showkeys(s).x2(1,inls12(1,:)) + size(I1,2) + 10;
  y2 = scale2*showkeys(s).y2(1,inls12(1,:));
  
  mh = scatter([x1'; x2'], [y1'; y2']);
  set(mh, 'MarkerEdgeColor', 'g', 'MarkerFaceColor', 'g', 'SizeData', 10);
  
  lh = line([x1; x2], [y1; y2]);
  set(lh, 'Color', showkeys(s).color, 'LineStyle', showkeys(s).linestyle, 'LineWidth', showkeys(s).linewidth);
  
end


end

