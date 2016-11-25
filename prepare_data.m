function prepare_data(dbname, makeFigure)
if ~exist('dbname', 'var')
  dbname = '20160601';
end

if ~exist('makeFigure', 'var')
  makeFigure = 0;
end

% -- Database info
datasize = 50;
p_no = 6;
scale_mult = 1;


% Labels:
%   0 - FirstPin, 
%   1 - B1, 
%   2 - B2, 
%   3 - B3, 
%   4 - B4, 
%   5 - SecondPin

datadir = sprintf('./dataset/%s/',dbname);
count = 1;

for i = 0:datasize-1
  fprintf('processing %d | %d\n', i, datasize);
  filepath = sprintf('./dataset/%s/resultPoint/%d-1.txt', dbname, i);
  annofile = fopen(filepath, 'r');
  C = textscan(annofile,'%s %s %s %s %s %s', 'Delimiter','\t');
  num_obj = length(C{1});
  
  for j = 1:num_obj
    % parse locations
    anno = struct('x', cell(p_no, 1), 'y', cell(p_no, 1));
    for p = 1:p_no
      [anno(p).x, anno(p).y] = strread(C{p}{j}(2:end-1), '%d, %d');
      joint_all(count).joint_self(p, 1) = anno(p).x;
      joint_all(count).joint_self(p, 2) = anno(p).y;
      joint_all(count).joint_self(p, 3) = 1;  % visibility
      if anno(p).x == -1 && anno(p).y == -1
        joint_all(count).joint_self(p, 3) = 0;  % visibility
      end
    end
    % add center point
    visible_parts = find(joint_all(count).joint_self(:, 3) ~= 0);
    cent.x = mean(joint_all(count).joint_self(visible_parts, 1));
    cent.y = mean(joint_all(count).joint_self(visible_parts, 2));
    joint_all(count).joint_self(p_no+1, 1) = cent.x;
    joint_all(count).joint_self(p_no+1, 2) = cent.y;
    joint_all(count).joint_self(p_no+1, 3) = 1;
    
    
    % reformat
    joint_all(count).dataset = 'shortPin';
    joint_all(count).isValidation = 0;
    
    % set image path
    joint_all(count).img_paths = sprintf('%d.bmp', i);
    [h,w,~] = size(imread([datadir 'originImg/' , joint_all(count).img_paths]));
    joint_all(count).img_width = w;
    joint_all(count).img_height = h;
    
    
    joint_all(count).objpos = [cent.x cent.y];
    
    % set scale
    if joint_all(count).joint_self(2, 3) && joint_all(count).joint_self(4, 3)
      diag = sqrt((anno(2).x - anno(4).x)^2 + (anno(2).y - anno(4).y)^2);
      joint_all(count).scale_provided = diag*scale_mult;
    else
      joint_all(count).scale_provided = 130; % average
    end
    
    % for other person on the same image
    count_other = 1;
    joint_others = cell(0,0);
    for op = 1:num_obj
      if(op == j), continue; end
      joint_others{count_other} = zeros(p_no,3);
      % read other joints
      anno_op = struct('x', cell(p_no, 1), 'y', cell(p_no, 1));
      for p = 1:p_no
        [anno_op(p).x, anno_op(p).y] = strread(C{p}{op}(2:end-1), '%d, %d');
        joint_all(count).joint_others{count_other}(p, 1) = anno_op(p).x;
        joint_all(count).joint_others{count_other}(p, 2) = anno_op(p).y;
        joint_all(count).joint_others{count_other}(p, 3) = 1;
        if anno(p).x == -1 && anno(p).y == -1
          joint_all(count).joint_self(p, 3) = 0;  % visibility
        end
      end
      % add center point
      visible_parts = find(joint_all(count).joint_others{count_other}(:, 3) ~= 0);
      cent.x = mean(joint_all(count).joint_others{count_other}(visible_parts, 1));
      cent.y = mean(joint_all(count).joint_others{count_other}(visible_parts, 2));
      joint_all(count).joint_others{count_other}(p_no+1, 1) = cent.x;
      joint_all(count).joint_others{count_other}(p_no+1, 2) = cent.y;
      joint_all(count).joint_others{count_other}(p_no+1, 3) = 1;
      
      % set scale
      
    if joint_all(count).joint_others{count_other}(2, 3) && joint_all(count).joint_others{count_other}(4, 3)
      diag = sqrt((anno_op(2).x - anno_op(4).x)^2 + (anno_op(2).y - anno_op(4).y)^2);
      joint_all(count).scale_provided_other(count_other) = diag*scale_mult;
    else
      joint_all(count).scale_provided_other(count_other) = 130; % in average
    end
    
      joint_all(count).objpos_other{count_other} = 0.5*[anno_op(2).x+anno_op(4).x, anno_op(2).y+anno_op(4).y];
      
      count_other = count_other + 1;
    end
    
    if(makeFigure) % visualizing to debug
      imshow([datadir 'originImg/', joint_all(count).img_paths]);
      hold on;
      visiblePart = joint_all(count).joint_self(1:p_no,3) == 1;
      invisiblePart = joint_all(count).joint_self(1:p_no,3) == 0;
      plot(joint_all(count).joint_self(visiblePart, 1), joint_all(count).joint_self(visiblePart,2), 'g.', 'MarkerSize', 20);
      plot(joint_all(count).joint_self(invisiblePart,1), joint_all(count).joint_self(invisiblePart,2), 'r.', 'MarkerSize', 20);
      plot(joint_all(count).objpos(1), joint_all(count).objpos(2), 'cs');
      if(~isempty(joint_all(count).joint_others))
        for op = 1:length(joint_all(count).joint_others)
          visiblePart = joint_all(count).joint_others{op}(:,3) == 1;
          invisiblePart = joint_all(count).joint_others{op}(:,3) == 0;
          plot(joint_all(count).joint_others{op}(visiblePart,1), joint_all(count).joint_others{op}(visiblePart,2), 'm.', 'MarkerSize', 20);
          plot(joint_all(count).joint_others{op}(invisiblePart,1), joint_all(count).joint_others{op}(invisiblePart,2), 'c.', 'MarkerSize', 20);
        end
      end
      pause;
      close all;
    end
    joint_all(count).annolist_index = i;
    joint_all(count).people_index = p;
    joint_all(count).numOtherPeople = length(joint_all(count).joint_others);
    count = count + 1;
    
  end
  
end
opt.FileName = ['cache/json/' dbname '.json'];
opt.FloatFormat = '%.3f';
savejson('root', joint_all, opt);
save(['cache/json/' dbname '.mat'], 'joint_all');

