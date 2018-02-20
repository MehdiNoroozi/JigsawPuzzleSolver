function [data, label] = parse_value(value)
% set_mode_cpu()
%   set Caffe to CPU mode

[data, label] = caffe_('parse_value', value);

end
