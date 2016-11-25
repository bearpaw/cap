function prepare_data(dbname)
datadir = sprintf('./dataset/%s/',dbname);
p_no = 6;
scale_mult = 1;

imlist = dir([datadir 'originImg/*.jpg']);
isValidation = zeros(length(imlist), 1);

s = RandStream('mt19937ar','Seed',0);

valid_list = randperm(s, length(imlist), floor(length(imlist)*0.2));
isValidation(valid_list) = 1;
% in Pin: (0 - FirstPin, 1 - B1, 2 - B2, 3 - B3, 4 - B4, 5 - SecondPin

count = 1;
makeFigure = 0;
validationCount = 0;

for i = 1:length(imlist)
  [~, name, ext] = fileparts(imlist(i).name);
  fprintf('processing %d | %d: %s\n', i, length(imlist), name);
  filepath = [datadir 'resultPoint/' name '.bmp.txt'];
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
    end
    % add center point
    cent.x = round(0.5*(max(joint_all(count).joint_self(:, 1)) + min(joint_all(count).joint_self(:, 1))));
    cent.y = round(0.5*(max(joint_all(count).joint_self(:, 2)) + min(joint_all(count).joint_self(:, 2))));
    joint_all(count).joint_self(p_no+1, 1) = cent.x;
    joint_all(count).joint_self(p_no+1, 2) = cent.y;
    joint_all(count).joint_self(p_no+1, 3) = 1;
    
    
    % reformat
    joint_all(count).dataset = 'shortPin';
    joint_all(count).isValidation = isValidation(i);
    
    % set image path
    joint_all(count).img_paths = imlist(i).name;
    [h,w,~] = size(imread([datadir 'originImg/' , joint_all(count).img_paths]));
    joint_all(count).img_width = w;
    joint_all(count).img_height = h;
    joint_all(count).objpos = 0.5*[anno(2).x+anno(4).x, anno(2).y+anno(4).y];
    % set part label: joint_all is (np-3-nTrain)
    
    % set scale
    diag = sqrt((anno(2).x - anno(4).x)^2 + (anno(2).y - anno(4).y)^2);
    joint_all(count).scale_provided = diag*scale_mult;
    
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
      end
      % add center point
      cent.x = round(0.5*(max(joint_all(count).joint_others{count_other}(:, 1)) + min(joint_all(count).joint_others{count_other}(:, 1))));
      cent.y = round(0.5*(max(joint_all(count).joint_others{count_other}(:, 2)) + min(joint_all(count).joint_others{count_other}(:, 2))));
      joint_all(count).joint_others{count_other}(p_no+1, 1) = cent.x;
      joint_all(count).joint_others{count_other}(p_no+1, 2) = cent.y;
      joint_all(count).joint_others{count_other}(p_no+1, 3) = 1;
      
      % set scale
      diag = sqrt((anno_op(2).x - anno_op(4).x)^2 + (anno_op(2).y - anno_op(4).y)^2);
      joint_all(count).scale_provided_other(count_other) = diag*scale_mult;
      joint_all(count).objpos_other{count_other} = 0.5*[anno_op(2).x+anno_op(4).x, anno_op(2).y+anno_op(4).y];
      
      count_other = count_other + 1;
    end
    
    if(makeFigure) % visualizing to debug
      imshow([datadir 'originImg/', joint_all(count).img_paths]);
      hold on;
      visiblePart = joint_all(count).joint_self(:,3) == 1;
      invisiblePart = joint_all(count).joint_self(:,3) == 0;
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
    
    %if(count==10), break; endscale_provided
  end
  %if(count==10), break; end
end
opt.FileName = ['cache/json/' dbname '.json'];
opt.FloatFormat = '%.3f';
savejson('root', joint_all, opt);
save(['cache/json/' dbname '.mat'], 'joint_all');

