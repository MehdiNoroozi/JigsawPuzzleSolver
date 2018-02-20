function mean_val = read_mean(file_path)

mean_val = caffe_('read_mean', file_path);

end
