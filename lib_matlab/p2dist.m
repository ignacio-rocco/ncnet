function [dpos,dori] = p2dist(P1,P2)


c1 = p2c(P1);
c2 = p2c(P2);

dpos = sqrt(sum((c1 - c2).^2));

R = P1(1:3,1:3)\P2(1:3,1:3);
dori = acos((trace(R)-1)/2);


% r1 = vl_irodr(P1(1:3,1:3));
% r2 = vl_irodr(P2(1:3,1:3));
% 
% % dori = acos((trace(P1(1:3,1:3) * P2(1:3,1:3)') - 1)/2);
% 
% dori = acos((r1./vnorm(r1))'*(r2./vnorm(r2)));
% 
% % v1 = P1(1:3,1:3)*[0;0;1];
% % v2 = P2(1:3,1:3)*[0;0;1];
% % dori = acos(v1'*v2);
% 
% keyboard;