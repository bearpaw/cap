function test_model_20160923_demo
% Change log
%   - 2016-08-01: more strict parallel constraints
%   - 2016-09-23: comments
close all;
cachedir = 'cache/20160801';
% -------------------------------------------------------------------------
% Compute distance between each pair of keypoints [for postprocessing]
% -------------------------------------------------------------------------
try
  load('cache/joint_dist.mat', 'means', 'vars');
catch
  load cache/json/20160601.mat
  joint_self = reshape([joint_all.joint_self], 7, 3, 2769);
  np = 6;
  
  means = zeros(np, np);
  vars = zeros(np, np);
  
  for i = 1:6
    for j = 1:6
      if i == j; continue; end
      valid = intersect( find(joint_self(i, 3, :) == 1), find(joint_self(j, 3, :) == 1));
      diff = squeeze( joint_self(i, 1:2, valid)-joint_self(j, 1:2, valid));
      dist = sqrt(sum(diff.^2, 1));
      means(i, j) = mean(dist);
      vars(i, j) = var(dist);
    end
  end
  save('cache/joint_dist.mat', 'means', 'vars');
end

% -------------------------------------------------------------------------
% initialize network
% -------------------------------------------------------------------------
iter = 80000;
dbname = '20160616';

caffemodel = sprintf('./prototxt/%s/caffemodel/pose_iter_%d.caffemodel',dbname, iter);
deployFile = sprintf('./prototxt/%s/pose_deploy_stage1.prototxt', dbname);
deployFile = sprintf('./prototxt/%s/pose_deploy_stage1.prototxt', dbname);

caffepath = '/home/wyang/code/caffe-bearpaw/matlab';
addpath(caffepath);
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(2);

net = caffe.Net(deployFile, caffemodel, 'test');

% Visualize the feature of network for debugging
if 1
  feat = net.params('conv1_stage1', 1).get_data();
  display_network_4D(feat);
  disp('visualize network. Press anykey to continue...');
  pause; close
end


% -------------------------------------------------------------------------
% Testing
% -------------------------------------------------------------------------
datadir = sprintf('./dataset/%s', dbname);
scale_search = 1; % 0.7:0.1:1.3;     % multi-scale testing
np = 6;                         % number of keypoints
testidx = 50:99;                % test images

target_dist=160*368/1200;       % matching the training image scale
boxsize = 368;                  % input cropsize

imdirname = 'rawImage';
visdirname = 'outImage';

for idx = testidx
  tic;
  imagePath = sprintf('%s/%s/%d.bmp',  datadir, imdirname, idx);
  
  try
    oriImg = imread(imagePath);
    if size(oriImg, 1) ~= 1024
      oriImg = imresize(oriImg, [1024, 1280]);
    end
    visImg = oriImg;
  catch
    error('image cannot be loaded, make sure you have %s', imagePath);
  end
  
  % --- compute center and original scale
  center = [size(oriImg,2), size(oriImg,1)]/2;
  scale_provided = 150;
  scale0 = target_dist/scale_provided;
  
  % --- get multiscale scoremaps
  multiplier = scale_search;
  score = cell(1, length(multiplier));
  pad = cell(1, length(multiplier));
  
  for m = 1:length(multiplier)
    % prepare data
    scale = scale0 * multiplier(m);
    imageToTest = imresize(oriImg, scale);
    
    center_s = center * scale;
    [imageToTest, pad{m}] = padAround(imageToTest, boxsize, center_s);
    
    %     imageToTest = preprocess(imageToTest, 0.5, center_map);
    imageToTest = preprocess(imageToTest, 0.5);
    % CNN
    score{m} = applyDNN(imageToTest, net);
    pool_time = size(imageToTest,1) / size(score{m},1);
    score{m} = imresize(score{m}, pool_time);
    score{m} = resizeIntoScaledImg(score{m}, pad{m});
    score{m} = imresize(score{m}, [size(oriImg,2), size(oriImg,1)]);
  end
  
  % --- summing up scores (fusing multiscale results)
  final_score = zeros(size(score{1,1}));
  for m = 1:size(score,2)
    final_score = final_score + score{m};
  end
  final_score = permute(final_score, [2 1 3]);
  
  % --- generate prediction
  prediction = zeros(np,2);
  for j = 1:np
    [prediction(j,2), prediction(j,1)] = findMaximum(final_score(:,:,j));
  end
  forwardtime = toc;
  
  %% Post Processing
  % --- Step #0: Find peaks from scoremaps
  tic;
  minidx = zeros(np,1)-1;
  peaks = cell(1, 6);
  for j = 1:np
    % Compute local peaks
    norm_score = normalize_score(final_score(:,:,j));
    thresh = uint16( 0.4*2^16);
    thresh = uint16( 0.2*2^16);
    filt = fspecial('gaussian', 21,1);
    p=FastPeakFind(norm_score, thresh);
    peaks{j} = reshape(p, 2, length(p)/2);
  end
  
  % --- Step #1: find the first edge (3, 4)
  cnt = 1;
  for j = 1:length(peaks{3})
    a = peaks{3}(:, j)';
    for k = 1:length(peaks{4})
      b = peaks{4}(:, k)';
      diff = a - b;
      dist = sqrt(sum(diff.^2));
      if abs(dist - means(3, 4)) < vars(3, 4)
        edgepairs(cnt).a = a;
        edgepairs(cnt).b = b;
        cnt = cnt + 1;
      end
    end
  end
  
  % --- Step #2: find the rest of edges. If the edge cannot be found, skip
  obj = 1;
  for cnt = 1:length(edgepairs)    
    % positive side
    A = edgepairs(cnt).a;
    B = edgepairs(cnt).b;
    x = [A(1);B(1)];
    y = [A(2);B(2)];
    
    curobj = [];
    for multiplier = [1, -1]
      % --- Step #2.1: find edge with keypoint [3, 2]
      pt32 = A + null(A-B)'*multiplier*(means(3, 2));
      candidate32 = [];
      mindist = 1000000;
      for j = 1:length(peaks{2})
        dist = sqrt(sum((pt32 - peaks{2}(:, j)').^2));
        if dist < vars(3, 2)
          if isempty( candidate32 )
            candidate32 = peaks{2}(:, j)';
            mindist = dist;
          else
            if dist < mindist
              candidate32 = peaks{2}(:, j)';
              mindist = dist;
            end
          end
        end
      end
      
      if isempty(candidate32)
        continue;
      end
      
      
      % --- Step #2.2: find edge with keypoint [4,5]
      pt45 = B + null(A-B)'*multiplier*(means(4, 5));
      candidate45 = [];
      mindist = 1000000;
      for j = 1:length(peaks{5})
        dist = sqrt(sum((pt45 - peaks{5}(:, j)').^2));
        if dist < vars(3, 2)
          if isempty( candidate45 )
            candidate45 = peaks{5}(:, j)';
            mindist = dist;
          else
            if dist < mindist
              candidate45 = peaks{5}(:, j)';
              mindist = dist;
            end
          end
        end
      end
      
      if isempty(candidate45)
        continue;
      end
      
      % --- Step #2.3: find edge with keypoint [2, 5]
      if abs(sqrt(sum((candidate45 -candidate32).^2)) - means(2, 5)) >= vars(2, 5)
        continue;
      end
      
      slope54 = (candidate45(2) - B(2))/(candidate45(1) - B(1));
      slope23 = (candidate32(2) - A(2))/(candidate32(1) - A(1));
      
      slope34 = (A(2) - B(2))/(A(1) - B(1));
      slope25 = (candidate32(2) - candidate45(2))/(candidate32(1) - candidate45(1));
      
      if abs(slope25 - slope34) > 0.5 || abs(slope54 - slope23) > 0.5
        continue;
      end
      
      % --- Step #2.4: find the constraint of pin 1
      mid25 = (candidate32+candidate45)/2;
      pt21 = mid25 + null(A-B)'*multiplier*(means(2, 1));
      candidate21 = [];
      mindist = 1000000;
      for j = 1:length(peaks{1})
        dist = sqrt(sum((pt21 - peaks{1}(:, j)').^2));
        if dist < vars(2, 1)
          if isempty( candidate21 )
            candidate21 = peaks{1}(:, j)';
            mindist = dist;
          else
            if dist < mindist
              candidate21 = peaks{1}(:, j)';
              mindist = dist;
            end
          end
        end
      end
      
      if isempty(candidate21)
        continue;
      end
      
      % --- Step #2.5: find the constraint of pin 6
      pt56 = mid25 + null(A-B)'*multiplier*(means(2, 1));
      candidate56 = [];
      mindist = 1000000;
      for j = 1:length(peaks{6})
        dist = sqrt(sum((pt56 - peaks{6}(:, j)').^2));
        if dist < vars(2, 1)
          if isempty( candidate56 )
            candidate56 = peaks{6}(:, j)';
            mindist = dist;
          else
            if dist < mindist
              candidate56 = peaks{6}(:, j)';
              mindist = dist;
            end
          end
        end
      end
      
      if isempty(candidate56)
        continue;
      end
      
      
      joints = [candidate21; candidate32; A; B; candidate45; candidate56];
      [jscore, score] = object_score(joints, final_score);
      joints = [joints, jscore];
      
      %--------------------------------------------------------------------
      %    Refine Joints
      %--------------------------------------------------------------------
      joints = refine_joints(joints);
      
      if isempty(curobj)
        curobj.joints = joints;
        curobj.score = score;
      elseif score > curobj.score
        curobj.joints = joints;
        curobj.score = score;
      end
    end
    if ~isempty(curobj)
      objects(obj).joints = curobj.joints ;
      objects(obj).score = curobj.score;
      obj = obj + 1;
    end
  end
  
  posttime = toc;
  fprintf('image %d/%d | forward: %.2fs | Post: %.2fs\n', ...
    idx, length(testidx), forwardtime, posttime);
  
  % show results
  clf;
  imshow(oriImg); hold on;
  
  for o = 1:obj-1
    joints = objects(o).joints;
    score = objects(o).score;
    
    line([joints(2, 1), joints(3, 1)],[joints(2, 2), joints(3, 2)],'color','r','LineWidth',3); hold on;
    line([joints(3, 1), joints(4, 1)],[joints(3, 2), joints(4, 2)],'color','r','LineWidth',3); hold on;
    line([joints(4, 1), joints(5, 1)],[joints(4, 2), joints(5, 2)],'color','r','LineWidth',3); hold on;
    line([joints(5, 1), joints(2, 1)],[joints(5, 2), joints(2, 2)],'color','r','LineWidth',3); hold on;
    
    line([mean(joints(2:5, 1)), joints(1, 1)],[mean(joints(2:5, 2)), joints(1, 2)],'color','c','LineWidth',2); hold on;
    line([mean(joints(2:5, 1)), joints(6, 1)],[mean(joints(2:5, 2)), joints(6, 2)],'color','y','LineWidth',2); hold on;
    text(mean(joints(2:5, 1)), mean(joints(2:5, 2)), sprintf('%.2f', score), 'FontSize', 36, 'Color', 'r', 'FontWeight', 'bold');
    
    
    
    markersize = 20;
    plot(joints(2, 1), joints(2, 2), 'c.', 'MarkerSize', markersize); hold on;
    plot(joints(3, 1), joints(3, 2), 'y.', 'MarkerSize', markersize); hold on;
    plot(joints(4, 1), joints(4, 2), 'm.', 'MarkerSize', markersize); hold on;
    plot(joints(5, 1), joints(5, 2), 'g.', 'MarkerSize', markersize); hold on;
    
  end
  
  z=getframe(gcf);
  imwrite(imresize(z.cdata, [512 640]), sprintf([cachedir '/%d_%.2d.png'], idx, np+1));
  clf;
end
prediction_file = sprintf('cache/%s.mat', dbname);
save(prediction_file, 'prediction_all');




function img_out = preprocess(img, mean, center_map)
img = double(img) / 256;

img_out = double(img) - mean;
img_out = permute(img_out, [2 1 3]);
img_out = img_out(:,:,[3 2 1]);

if exist('center_map', 'var')
  img_out(:,:,4) = center_map{1};
end

function scores = applyDNN(images, net)
input_data = {single(images)};
% do forward pass to get scores
% scores are now Width x Height x Channels x Num
scores = net.forward(input_data);
%     scores = net.blobs('Fconv7_stage1').get_data();
scores = scores{1};

function [img_padded, pad] = padAround(img, boxsize, center)
center = round(center);
h = size(img, 1);
w = size(img, 2);
pad(1) = boxsize/2 - center(2); % up
pad(3) = boxsize/2 - (h-center(2)); % down
pad(2) = boxsize/2 - center(1); % left
pad(4) = boxsize/2 - (w-center(1)); % right

pad_up = repmat(img(1,:,:)*0, [pad(1) 1 1])+128;
img_padded = [pad_up; img];
pad_left = repmat(img_padded(:,1,:)*0, [1 pad(2) 1])+128;
img_padded = [pad_left img_padded];
pad_down = repmat(img_padded(end,:,:)*0, [pad(3) 1 1])+128;
img_padded = [img_padded; pad_down];
pad_right = repmat(img_padded(:,end,:)*0, [1 pad(4) 1])+128;
img_padded = [img_padded pad_right];

center = center + [max(0,pad(2)) max(0,pad(1))];
img_padded = img_padded(center(2)-(boxsize/2-1):center(2)+boxsize/2, center(1)-(boxsize/2-1):center(1)+boxsize/2, :); %cropping if needed

function [x,y] = findMaximum(map)
[~,i] = max(map(:));
[x,y] = ind2sub(size(map), i);

function score = resizeIntoScaledImg(score, pad)
np = size(score,3)-1;
score = permute(score, [2 1 3]);
if(pad(1) < 0)
  padup = cat(3, zeros(-pad(1), size(score,2), np), ones(-pad(1), size(score,2), 1));
  score = [padup; score]; % pad up
else
  score(1:pad(1),:,:) = []; % crop up
end

if(pad(2) < 0)
  padleft = cat(3, zeros(size(score,1), -pad(2), np), ones(size(score,1), -pad(2), 1));
  score = [padleft score]; % pad left
else
  score(:,1:pad(2),:) = []; % crop left
end

if(pad(3) < 0)
  paddown = cat(3, zeros(-pad(3), size(score,2), np), ones(-pad(3), size(score,2), 1));
  score = [score; paddown]; % pad down
else
  score(end-pad(3)+1:end, :, :) = []; % crop down
end

if(pad(4) < 0)
  padright = cat(3, zeros(size(score,1), -pad(4), np), ones(size(score,1), -pad(4), 1));
  score = [score padright]; % pad right
else
  score(:,end-pad(4)+1:end, :) = []; % crop right
end
score = permute(score, [2 1 3]);

% function headSize = util_get_head_size(rect)
%     SC_BIAS = 0.6; % 0.8*0.75
%     headSize = SC_BIAS * norm([rect.x2 rect.y2] - [rect.x1 rect.y1]);

function bodysize = util_get_bodysize_size(rect)
bodysize = norm(rect(10,:) - rect(3,:)); % following evalLSP_official

function label = produceCenterLabelMap(im_size, x, y, sigma) %this function is only for center map in testing
[X,Y] = meshgrid(1:im_size(1), 1:im_size(2));
X = X - x;
Y = Y - y;
D2 = X.^2 + Y.^2;
Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
label{1} = exp(-Exponent);

function plot_box(centers, bsize)
% centers should be 2xn
num = size(centers, 2);
hsize = round(bsize/2);
for i = 1:num
  x1 = centers(1, i) - hsize;
  y1 = centers(2, i) - hsize;
  x2 = centers(1, i) + hsize - 1;
  y2 = centers(2, i) + hsize - 1;
  plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1]);
  hold on;
end

function score = normalize_score( score )
score = (score - min(score(:)))/ (max(score(:)) - min(score(:)));

function [joint_score, score] = object_score(joints, score_map)
score = 0;
joint_score = zeros(6, 1);
minloc = min(joints);
maxloc = max(joints);

if minloc(1) < 1 || minloc(2)<1 || maxloc(1) > size(score_map, 1) || maxloc(2) > size(score_map, 2)
  score = 0;
else
  for j = 1:size(joints, 1)
    joint_score(j) = score_map(joints(j, 1), joints(j, 2), j);
    score = score + joint_score(j);
  end
  score = score / size(joints, 1);
end


function joints = refine_joints(joints)
% -------------------------------------------------------------------------
% Refine the joints by a strong parrellel and perpendicular relationship
%   input 
%       - joints: [npointx3] matrix [x, y, score]
%   output
%       - joints: the refined joints
% -------------------------------------------------------------------------
jscore = joints(:, 3);
joints = joints(:, 1:2);
vertical_mid = [mean(joints(2:3, :)); mean(joints(4:5, :))];
horizont_mid = [mean([joints(2, :); joints(5, :)]); 
                mean(joints(3:4, :))];
[val, ind] = max([mean(jscore(2:3)),mean(jscore(3:4)),mean(jscore(4:5)),mean([jscore(2),jscore(5)])]);


% case 1
if ind == 1
  slope1 = (joints(2, 2) - joints(3, 2))/(joints(2, 1) - joints(3, 1));
  slope2 = -1/slope1;

  joints(5, :) = linecross(slope1, vertical_mid(2, :), slope2, horizont_mid(1, :));
  joints(4, :) = linecross(slope1, vertical_mid(2, :), slope2, horizont_mid(2, :));
  joints(2, :) = linecross(slope1, vertical_mid(1, :), slope2, horizont_mid(1, :));
  joints(3, :) = linecross(slope1, vertical_mid(1, :), slope2, horizont_mid(2, :));
elseif ind == 2
  slope1 = (joints(3, 2) - joints(4, 2))/(joints(3, 1) - joints(4, 1));
  slope2 = -1/slope1;

  joints(3, :) = linecross(slope1, horizont_mid(2, :), slope2, vertical_mid(1, :));
  joints(4, :) = linecross(slope1, horizont_mid(2, :), slope2, vertical_mid(2, :));
  joints(2, :) = linecross(slope1, horizont_mid(1, :), slope2, vertical_mid(1, :));
  joints(5, :) = linecross(slope1, horizont_mid(1, :), slope2, vertical_mid(2, :));
elseif ind == 3
  slope1 = (joints(4, 2) - joints(5, 2))/(joints(4, 1) - joints(5, 1));
  slope2 = -1/slope1;

  joints(5, :) = linecross(slope1, vertical_mid(2, :), slope2, horizont_mid(1, :));
  joints(4, :) = linecross(slope1, vertical_mid(2, :), slope2, horizont_mid(2, :));
  joints(2, :) = linecross(slope1, vertical_mid(1, :), slope2, horizont_mid(1, :));
  joints(3, :) = linecross(slope1, vertical_mid(1, :), slope2, horizont_mid(2, :));
else
  slope1 = (joints(2, 2) - joints(5, 2))/(joints(2, 1) - joints(5, 1));
  slope2 = -1/slope1;

  joints(3, :) = linecross(slope1, horizont_mid(2, :), slope2, vertical_mid(1, :));
  joints(4, :) = linecross(slope1, horizont_mid(2, :), slope2, vertical_mid(2, :));
  joints(2, :) = linecross(slope1, horizont_mid(1, :), slope2, vertical_mid(1, :));
  joints(5, :) = linecross(slope1, horizont_mid(1, :), slope2, vertical_mid(2, :));
end
joints = [joints, jscore];

% plot(x(1), x(2), 'r.', 'MarkerSize', 20); hold on;
% 
% 
% point_slope_line(slope1, mean(joints(4:5, :)));
% point_slope_line(slope1, mean(joints(2:3, :)));
% point_slope_line(slope2, mean(joints(3:4, :)));
% point_slope_line(slope2,mean([joints(2, :); joints(5, :)]));
% 
% daspect([1 1 1])


function x = linecross(slope1, point1, slope2, point2)
% -------------------------------------------------------------------------
% Compute the cross point given two lines
%   input 
%       - SlopeX, pointX: the slope of the line, and one point on the line
%   output
%       - x: the [x, y] coordinate of the cross point
% -------------------------------------------------------------------------
A = [slope1, -1; slope2, -1];
b = [slope1*point1(1) - point1(2); 
  slope2*point2(1) - point2(2)];
x = A\b;


