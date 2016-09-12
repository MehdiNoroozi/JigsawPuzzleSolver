function [data, label, l] = read_next_batch(n)
% set_mode_cpu()
%   set Caffe to CPU mode

[data, label, l] = caffe_('read_next_batch', n);

end
