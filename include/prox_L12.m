function [wp] = prox_L12(wx,gamma)

[R,N] = size(wx);
wp    = zeros(R,N);

tmp         = sqrt(sum(wx.^2,1));
ind         = find(tmp>gamma);
wp(:,ind)   = (ones(R,1)*(1 - gamma./tmp(ind))).*wx(:,ind);


