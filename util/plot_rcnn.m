function plot_box(boxes, LineWidth)
% box: x1, y1, x2, y2, score
if ~exist('LineWidth', 'var')
  LineWidth = 4;
end
nboxes = size(boxes, 1);

for i = 1:nboxes
  x1 = boxes(i, 1);
  y1 = boxes(i, 2);
  x2 = boxes(i, 3);
  y2 = boxes(i, 4);
  score = boxes(i, 5);
  if score < 0.95
    continue;
  end
  % plot
  plot([x1 x1 x2 x2 x1], [y1 y2 y2 y1 y1], 'LineWidth', LineWidth);
%   text( (x1+x2)/2, (y1+y2)/2, sprintf('%.2f', score));
end