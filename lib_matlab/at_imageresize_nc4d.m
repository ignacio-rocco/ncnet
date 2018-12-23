function im1 = at_imageresize_nc4d(im1,imax)
if nargin < 2, imax = 1920; end

isz = size(im1(:,:,1));
if max(isz) > imax
  if isz(1) > isz(2)
    im1 = imresize(im1,[imax NaN]);
  else
    im1 = imresize(im1,[NaN imax]);
  end
end


% isz = size(im1(:,:,1));
% if (1920*1440) < prod(isz)
%   if isz(1) > isz(2)
%     im1 = imresize(im1,[1920 NaN]);
%   else
%     im1 = imresize(im1,[NaN 1920]);
%   end
% end

% isz = size(im1(:,:,1));
% if (1600*1200) < prod(isz)
%   scale = 1600/max(isz);
%   im1 = imresize(im1, scale);
% %   if isz(1) > isz(2)
% %     im1 = imresize(im1,[640 NaN]);
% %   else
% %     im1 = imresize(im1,[NaN 640]);
% %   end
% end

% isz = size(im1(:,:,1));
% if (1920*1440) < prod(isz)
%   im1 = imresize(im1,sqrt((1920*1440)/prod(isz)));
% end
