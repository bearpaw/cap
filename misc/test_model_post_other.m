function test_model_post_other
%% initialize network
iter = 27000;
caffemodel = sprintf('./prototxt/shortPin_longPin_all/caffemodel/pose_iter_%d.caffemodel', iter);
deployFile = './prototxt/shortPin_longPin_all/pose_deploy_stage1.prototxt';

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
datadir = '/home/wyang/Data/dataset/robotics/2016041201/02/';
cachedir = './cache/may16/2016041201/02/';

if ~exist(cachedir, 'dir')
  unix(sprintf('mkdir -p %s', cachedir));
end

makeFigure = 1;
colors = colormap(lines(20)); close all;


scale_mult = 1;
batch_size = 4;
% scale_search = 0.7:0.1:1.3; % fit training
% scale_search = 0.7:0.1:0.9; % fit training
scale_search = 0.3:0.1:0.5; % fit training
np = 7;
center_channel = 7;
pa = [2, 3, 0, 3, 4, 5];
sigma = 21;
obj = zeros(1,np);
detected = zeros(1,np);

target_dist=160*368/1200;
boxsize = 368;

imlist = dir([datadir '/*.png']);

joint_all = cell(length(imlist), 1);

for i = 1:length(imlist)
  tic;
  imagePath = [datadir imlist(i).name];
  [~, imageName, ~] = fileparts(imagePath);
  try
    oriImg = imread(imagePath);
    oriImg = imresize(oriImg, 1600/size(oriImg, 2));
  catch
    error('image cannot be loaded, make sure you have %s', imagePath);
  end
  
  center = [size(oriImg,2), size(oriImg,1)]/2;
  %   scale_provided = size(oriImg,2)/boxsize; % something prop to image height
  scale_provided = 150;
  scale0 = target_dist/scale_provided;
  
  %   joint_gt(:,:,i) = joint_all(i).joint_self(:,1:2);
  
  multiplier = scale_search;
  score = cell(1, length(multiplier));
  pad = cell(1, length(multiplier));
  parts = struct('score', cell(1, length(multiplier)), ...
    'Ix', cell(1, length(multiplier)), 'Iy', cell(1, length(multiplier)));
  %   % ----------------------------------------------------------
  %   cnt = 1;
  %  for m = 1:batch_size:length(multiplier)
  %   num = min(batch_size, length(multiplier)-m+1);
  %   impyra = zeros(boxsize, boxsize, 3, num, 'single');
  %   for n = 0:num-1
  %         % prepare data
  %     scale = scale0 * multiplier(m);
  %     imageToTest = imresize(oriImg, scale);
  %     center_s = center * scale;
  %     [imageToTest, pad{m+n}] = padAround(imageToTest, boxsize, center_s);
  %     imageToTest = preprocess(imageToTest, 0.5);
  %
  %     impyra(:,:,:,n+1) = imageToTest;
  %   end
  %   if num == 1
  %     net.blobs('data').reshape([size(impyra) 1]);
  %   else
  %     net.blobs('data').reshape(size(impyra)); % reshape blob 'data'
  %   end
  %   net.reshape();
  %   resp = net.forward({impyra});      % softmax apply in caffe model.
  %   for n = 0:num-1
  %     score{m+n} = resp{1}(:,:,:,n+1);
  %     pool_time = size(imageToTest,1) / size(score{m+n},1);
  %     score{m+n} = imresize(score{m+n}, pool_time);
  %     score{m+n} = resizeIntoScaledImg(score{m+n}, pad{m+n});
  %     score{m+n} = imresize(score{m+n}, [size(oriImg,2), size(oriImg,1)]);
  %   end
  % end
  % ----------------------------------------------------------
  
  
  for m = 1:length(multiplier)
    % prepare data
    scale = scale0 * multiplier(m);
    imageToTest = imresize(oriImg, scale);
    
    center_s = center * scale;
    [imageToTest, pad{m}] = padAround(imageToTest, boxsize, center_s);
%     imshow(imageToTest); pause;
    %     imageToTest = preprocess(imageToTest, 0.5, center_map);
    imageToTest = preprocess(imageToTest, 0.5);
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
    [prediction(j,2), prediction(j,1)] = findMaximum(final_score(:,:,j));
  end
  %   prediction(order_to_lsp,:) = prediction;
  %   final_score(:,:,order_to_lsp) = final_score(:,:,1:np);
  elapse = toc;
  fprintf('image %d/%d | %.2f s\n', i, length(imlist), elapse);
  
  for j = 1:np
    
    % Compute local peaks
    norm_score = normalize_score(final_score(:,:,j));
    thresh = uint16( 0.4*2^16);
    filt = fspecial('gaussian', 21,1);
    p=FastPeakFind(norm_score, thresh);
    peaks{j} = reshape(p, 2, length(p)/2);
    
    
    
    
    if makeFigure %&& j == 3
      max_value = max(max(final_score(:,:,j)));
      if max_value == 0
        max_value = 1;
      end
      
      imToShow = single(oriImg)/255 * 0.5 + mat2im(final_score(:,:,j), jet(100), [0 max_value])/2;
      imToShow = insertShape(imToShow, 'FilledCircle', [prediction(j,:) 5], 'Color', 'w', 'Opacity', 1);
      %       imToShow = insertShape(imToShow, 'FilledCircle', [joint_gt(j,1:2,i) 2], 'Color', 'g');
      imToShow = insertShape(imToShow, 'FilledRectangle', [center 3 3], 'Color', 'c');
      imwrite(imresize(imToShow, 0.5), sprintf([cachedir '/%s_%.2d.jpg'], imageName, j));
%       imshow(imToShow); hold on;
%       plot_box(peaks{j}, 100);
%       title('paused, click to resume');
%       pause;
    end
  end
  
  num_det = size(peaks{center_channel}, 2);
  joint_all{i} = cell(num_det, 1);
  for obj = 1:num_det
    joint_all{i}{obj} = zeros(size(peaks{center_channel}));
    for p = 1:np
      if p == center_channel
        joint_all{i}{obj}(:, p) = peaks{center_channel}(:, obj);
        continue;
      end
      mindist = 1e10;
      minidx = -1;
      for peakid = 1:size(peaks{p}, 2)
        curdist = norm( peaks{p}(:, peakid) - peaks{center_channel}(:, obj));
        if curdist < mindist
          mindist = curdist;
          minidx = peakid;
        end
      end
      joint_all{i}{obj}(:, p) = peaks{p}(:, minidx);
    end
  end
  
  if makeFigure %&& j == 3
    imToShow = single(oriImg)/255;
    for obj = 1:num_det
      
      rectpos = joint_all{i}{obj}(:, 2:5);
      leftpin = [joint_all{i}{obj}(:, 1); joint_all{i}{obj}(:, center_channel)]';
      rightpin = [joint_all{i}{obj}(:, 6); joint_all{i}{obj}(:, center_channel)]';
      imToShow = insertShape(imToShow, 'Polygon', rectpos(:)', 'Color', colors(obj, :));
      imToShow = insertShape(imToShow, 'FilledPolygon', rectpos(:)', 'Color', colors(obj, :), 'Opacity', 0.2);
      imToShow = insertShape(imToShow, 'FilledCircle', [joint_all{i}{obj}(:, 1)' 6], 'Color', colors(obj, :), 'Opacity', 0.8);
      imToShow = insertShape(imToShow, 'FilledCircle', [joint_all{i}{obj}(:, 6)' 6], 'Color', colors(obj, :), 'Opacity', 0.8);
      imToShow = insertShape(imToShow, 'FilledRectangle', [joint_all{i}{obj}(:, center_channel)' 15 15], 'Color', 'w');
    end
    imwrite(imresize(imToShow, 0.5), sprintf([cachedir '%s_%.2d_visualize.jpg'], imageName, j));
    
%     imshow(imToShow); hold on;
%     title('paused, click to resume');
%     pause;
  end
  
  %   elapse = toc;
  %   fprintf('image %d/%d | %.2f s\n', i, length(imgnames), elapse);
  
  %   for j = 1:np
  %     fprintf(' %.3f', detected(j)/obj(j));
  %   end
  %   fprintf(' |%.4f\n', sum(detected)/sum(obj));
  %
  prediction_all(:,:,i) = prediction;
end
% save(sprintf('./cache/%s/prediction_all.mat', dbname), 'prediction_all');




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
