function write_mean(mean_data, mean_proto_file)

caffe_('write_mean', mean_data, mean_proto_file);

end
