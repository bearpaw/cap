addpath(genpath('./util'));
addpath(genpath('./mex_unix'));

if ~exist('cache', 'dir')
  mkdir('cache');
end


if ~exist('cache/json', 'dir')
  mkdir('cache/json');
end


if ~exist('cache/lmdb', 'dir')
  mkdir('cache/lmdb');
end

