function boxes = post_dt_process(im,iminfo,pyra,anchor)
INF = 1e10;
conf = global_conf();
pa = conf.pa;
p_no = 18;
w = [0.05,0,0.05,0];

full_bbx = round(iminfo.full_bbx); %  + [-psize, +psize]);
dim1 = size(im, 1); dim2 = size(im, 2);
full_bbx = [max(1, full_bbx(1)), max(1, full_bbx(2)), ...
    min(dim2, full_bbx(3)), min(dim1, full_bbx(4))];
iminfo.constr_bbx = iminfo.constr_bbx - [full_bbx(1), full_bbx(2), full_bbx(1), full_bbx(2)] + 1;

box = [];
boxes =[];
% boxes = zeros(length(pyra),p_no*4+1);
for l = 1:length(pyra)
    feat = pyra(l).feat(:,:,1:end-1);
    sizs= pyra(l).sizs;
    for i = 1:size(feat,3)
        parts(i).score = feat(:,:,i);
        if i==2
            constr_bbx = iminfo.constr_bbx;
            bbx2map = (constr_bbx - 1) ./ pyra(l).scale + 1;
            bbx2map = floor(bbx2map);
            bbx2map([1,2]) = max(1, bbx2map([1,2]));
            bbx2map([3,4]) = min([sizs(2), sizs(1)], bbx2map([3,4]));

            invalid_map = true(sizs(1), sizs(2));
            invalid_map(bbx2map(2):bbx2map(4), bbx2map(1):bbx2map(3)) = false;
            
            parts(i).score(invalid_map) = -INF;
        end
    end
    
    %-------- pass message -------------
    for i =  p_no:-1:2
        child.score = double(feat(:,:,i));
        child.w = w;
        child.startx  = anchor(i,1);
        child.starty  = anchor(i,2);
        parent.score = double((feat(:,:,pa(i))));
        [msg,Ix,Iy] = passmsg_my(child,parent);
%         [msg,Ix,Iy] = passmsg_my_multi(child,parent,anchor,13);
        
%         passmsg_my_multi
        parts(i).Ix = Ix;
        parts(i).Iy = Iy;
        parts(i).parent = pa(i);
        %         subplot(2,2,1);Transparent_overlay(im,parts(i).score,0.7);
        %         subplot(2,2,2);Transparent_overlay(im,msg,0.7);
        %         subplot(2,2,3);Transparent_overlay(im,parts(pa(i)).score,0.7);
        parts(pa(i)).score = parts(pa(i)).score+msg/5;
        %         subplot(2,2,4);Transparent_overlay(im,parts(pa(i)).score,0.7);
        %         text(1,1,sprintf('part%d',i));
    end
    
    
    parts(1).score = parts(1).score;
    [rscore Ik] = max(parts(1).score,[],3);
    
    A = sort(rscore(:));
    thresh = A(round(length(rscore(:))*0.95));
    
%     thresh = max(max(rscore));
    % Walk back down tree following pointers
    [Y,X] = find(rscore >= thresh);
    
    if length(X) >= 1,
        I   = (X-1)*size(rscore,1) + Y;
        box = backtrack_my(X,Y,Ik(I),parts,pyra(l));
        box = bsxfun(@plus, box, [full_bbx(1), full_bbx(2), full_bbx(1), full_bbx(2)]);
        box = box - 1;
        box = reshape(box,length(X),4*p_no);
    end
    boxes = [boxes; box rscore(I)];
    
end



