function test_model_20160723_demo
colors = colormap(lines(20)); close all;
cachedir = 'cache/20160724';
%% prepare post processing distances
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

%% initialize network
iter = 80000;
dbname = '20160616';

caffemodel = sprintf('./prototxt/%s/caffemodel/pose_iter_%d.caffemodel',dbname, iter);
deployFile = sprintf('./prototxt/%s/pose_deploy.prototxt', dbname);

caffepath = '/home/wyang/code/caffe-bearpaw/matlab';
addpath(caffepath);
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(2);

net = caffe.Net(deployFile, caffemodel, 'test');

% feat = net.params('Fconv7_stage1', 1).get_data();
%
% display_network_4D(feat);
% pause; close

%% settings
makeFigure = 1;
datadir = sprintf('./dataset/%s', dbname);
scale_mult = 1;
batch_size = 4;
% scale_search = 0.7:0.1:1.3; % fit training
% scale_search = 0.9:0.1:1.1; % fit training
scale_search = 0.8; % fit training
np = 6;
pa = [2, 3, 0, 3, 4, 5];
sigma = 21;
obj = zeros(1,np);
detected = zeros(1,np);

target_dist=160*368/1200;
boxsize = 368;

% load(['cache/json/' dbname '.mat'], 'joint_all');

imdirname = 'rawImage';
visdirname = 'outImage';

testidx = [50:99];
for idx = testidx
  tic;
  %   idx = i - 1;
  imagePath = sprintf('%s/%s/%d.bmp',  datadir, imdirname, idx);
  visimagePath = sprintf('%s/%s/%d-1.bmp',  datadir, visdirname, idx);
  
  [~, imageName, ~] = fileparts(imagePath);
  try
    oriImg = imread(imagePath);
    if size(oriImg, 1) ~= 1024
      oriImg = imresize(oriImg, [1024, 1280]);
    end
    visImg = oriImg;
  catch
    error('image cannot be loaded, make sure you have %s', imagePath);
  end
  
  % load faster RCNN box
  load(sprintf('cache/det50_99/%d.mat', idx)); % detection
  detection = detection(find(detection(:, 5)>0.97), :);
  
  % for each detection
  objects = [];
  for det = 1:size(detection, 1)
    %   center = [size(oriImg,2), size(oriImg,1)]/2;
    center = [detection(det, 1)+detection(det, 3),  detection(det, 2)+detection(det, 4)];
    center = center./2;
    %   scale_provided = size(oriImg,2)/boxsize; % something prop to image height
    scale_provided = 150;
    scale0 = target_dist/scale_provided;
    
    %   joint_gt(:,:,i) = joint_all(i).joint_self(:,1:2);
    
    multiplier = scale_search;
    score = cell(1, length(multiplier));
    pad = cell(1, length(multiplier));
    parts = struct('score', cell(1, length(multiplier)), ...
      'Ix', cell(1, length(multiplier)), 'Iy', cell(1, length(multiplier)));
    
    
    for m = 1:length(multiplier)
      % prepare data
      scale = scale0 * multiplier(m);
      imageToTest = imresize(oriImg, scale);
      
      center_s = center * scale;
      [imageToTest, pad{m}] = padAround(imageToTest, boxsize, center_s);
      
      center_map = produceCenterLabelMap([boxsize boxsize], boxsize/2, boxsize/2, 21);
      imageToTest = preprocess(imageToTest, 0.5, center_map);
      %     imageToTest = preprocess(imageToTest, 0.5);
      % CNN
      score{m} = applyDNN(imageToTest, net);
      pool_time = size(imageToTest,1) / size(score{m},1);
      score{m} = imresize(score{m}, pool_time);
      score{m} = resizeIntoScaledImg(score{m}, pad{m});
      score{m} = imresize(score{m}, [size(oriImg,2), size(oriImg,1)]);
    end
    
    % summing up scores
    final_score = zeros(size(score{1,1}));
    for m = 1:size(score,2)
      final_score = final_score + score{m};
    end
    final_score = permute(final_score, [2 1 3]);
    % generate prediction
    prediction = zeros(np,2);
    for j = 1:np
      %       clf; imagesc(final_score(:,:,j));pause;
      [prediction(j,2), prediction(j,1), score] = findMaximum(final_score(:,:,j));
    end
    objects(det).joints = prediction;
    %     objects(det).score = score
    objects(det).score = detection(det, 5);
  end
  
  elapse = toc;
  fprintf('image %d/%d | %.2f s\n', idx, length(testidx), elapse);
  
  
  
  % show results
  clf;
  imshow(oriImg); hold on;
  plot_rcnn(detection, 2);
  for o = 1:length(objects)
    joints = objects(o).joints;
    score = objects(o).score;
    
    line([joints(2, 1), joints(3, 1)],[joints(2, 2), joints(3, 2)],'color','r','LineWidth',3); hold on;
    line([joints(3, 1), joints(4, 1)],[joints(3, 2), joints(4, 2)],'color','r','LineWidth',3); hold on;
    line([joints(4, 1), joints(5, 1)],[joints(4, 2), joints(5, 2)],'color','r','LineWidth',3); hold on;
    line([joints(5, 1), joints(2, 1)],[joints(5, 2), joints(2, 2)],'color','r','LineWidth',3); hold on;
    
    line([mean(joints(2:5, 1)), joints(1, 1)],[mean(joints(2:5, 2)), joints(1, 2)],'color','c','LineWidth',2); hold on;
    line([mean(joints(2:5, 1)), joints(6, 1)],[mean(joints(2:5, 2)), joints(6, 2)],'color','y','LineWidth',2); hold on;
    text(mean(joints(2:5, 1)), mean(joints(2:5, 2)), sprintf('%.2f', score), 'FontSize', 36, 'Color', 'r', 'FontWeight', 'bold');
  end
%   pause;
  
  % visualize
%   for j = 1:np
%     max_value = max(max(final_score(:,:,j)));
%     imToShow = single(oriImg)/255 * 0.5 + mat2im(final_score(:,:,j), jet(100), [0 max_value])/2;
%     imwrite(imresize(imToShow, 0.5), sprintf([cachedir '/%d_%.2d.png'], idx, j));
%   end
  z=getframe(gcf);
  imwrite(imresize(z.cdata, [512 640]), sprintf([cachedir '/%d_%.2d.png'], idx, np+1));
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

function [x,y, score] = findMaximum(map)
[score,i] = max(map(:));
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

function score = object_score(joints, score_map)
score = 0;
minloc = min(joints);
maxloc = max(joints);

if minloc(1) < 1 || minloc(2)<1 || maxloc(1) > size(score_map, 1) || maxloc(2) > size(score_map, 2)
  score = 0;
else
  for j = 1:size(joints, 1)
    score = score + score_map(joints(j, 1), joints(j, 2), j);
  end
  score = score / size(joints, 1);
end
