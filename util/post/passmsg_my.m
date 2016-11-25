function [score0,Ix,Iy] = passmsg_my(child,parent)

INF = 1e10;

Ny  = size(parent.score,1);
Nx  = size(parent.score,2);
[Ix0,Iy0,score0] = deal(zeros([Ny Nx 1]));

k = 1;
[score0,Ix,Iy] = shiftdt_yy(child.score, ...
    child.w(1,k), child.w(1,k), child.w(1,k), child.w(1,k),...
    child.startx(k),child.starty(k),Nx,Ny,1);


% At each parent location, for each parent mixture 1:L, compute best child mixture 1:K
% L  = length(parent.filterid);