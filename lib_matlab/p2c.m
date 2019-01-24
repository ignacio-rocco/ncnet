function C = p2c(P)
% P = [R t], no K involved

C = -P(1:3,1:3)'*P(1:3,4);