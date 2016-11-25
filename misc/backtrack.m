% Backtrack through DP msgs to collect ptrs to part locations
function points = backtrack(x,y,parts,pyra)
  numx     = length(x);
  numparts = length(parts);
  
  xptr = zeros(numx,numparts);
  yptr = zeros(numx,numparts);
  points  = zeros(numx, 2, numparts);

  for k = 1:numparts,
    p   = parts(k);
    if k == 1,
      xptr(:,k) = x;
      yptr(:,k) = y;
    else
      % I = sub2ind(size(p.Ix),yptr(:,par),xptr(:,par),mptr(:,par));
      par = p.parent;
      [h,w,foo] = size(p.Ix);
      I   = (xptr(:,par)-1)*h + yptr(:,par);
      xptr(:,k) = p.Ix(I);
      yptr(:,k) = p.Iy(I);
    end
%     scale = pyra.scale(p.level);
%     x1 = (xptr(:,k) - 1 - pyra.padx)*scale+1;
%     y1 = (yptr(:,k) - 1 - pyra.pady)*scale+1;
%     x2 = x1 + p.sizx(mptr(:,k))*scale - 1;
%     y2 = y1 + p.sizy(mptr(:,k))*scale - 1;
%     box(:,:,k) = [x1 y1 x2 y2];
    points(:,:,k) = [xptr(:,k) yptr(:,k)];
  end
  points = reshape(points,numx,2*numparts);