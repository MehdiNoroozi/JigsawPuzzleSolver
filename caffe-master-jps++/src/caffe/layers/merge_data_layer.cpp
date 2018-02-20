#include <vector>

#include "caffe/layers/merge_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MergeDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  //const ConcatParameter& concat_param = this->layer_param_.concat_param();
  //CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
    //  << "Either axis or concat_dim should be specified; not both.";
}

template <typename Dtype>
void MergeDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const int num_axes = bottom[0]->num_axes();
  const MergeDataParameter& merge_data_param = this->layer_param_.merge_data_param();
  
  min_substitute_ = merge_data_param.min_substitute();
  max_substitute_ = merge_data_param.max_substitute();

  // Initialize with the first blob.
  vector<int> top_shape = bottom[0]->shape();
  //num_concats_ = bottom[0]->count(0, concat_axis_);
  //concat_input_size_ = bottom[0]->count(concat_axis_ + 1);
  
	for (int i = 0; i < top.size() - 1; ++i) 
	{
	  top[i]->Reshape(top_shape);
	}
	vector<int> top_shape_(2);// = top_shape;
	top_shape_[0] = top_shape[0];
	top_shape_[1] = 1;
	top[top.size() - 1]->Reshape(top_shape_);
	//printf("\n66666666666666666666666");
}

template <typename Dtype>
void MergeDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  //printf("\n--- " );
  int num_tiles = top.size() - 1;
  
  
  bool * copy_mask = new bool[num_tiles];
  int * rand_indices = new int[num_tiles];
  int copy_size = bottom[0]->count(1);
  printf("\n ^^^^^^^  %i, %i, %i^^^^^^^^^^^^", copy_size, bottom[0]->num(), num_tiles  );
  Dtype * sub_num_top = top[top.size()-1]->mutable_cpu_data();  
for( int n = 0; n < bottom[0]->num(); n++ )
  {
  int num_sub = rand()%max_substitute_ + min_substitute_;
  for( int i = 0; i < num_tiles;  i++ ) { rand_indices[i] = -1;  copy_mask[i] = false; }
  
  for( int i = 0; i < num_sub; i++ )
  {
	  int ind = 0;
	  while(1) { 
		   ind = rand()%num_tiles;
		  if( copy_mask[ind] == false ){copy_mask[ind] = true; break;}
	  }
	  int ind2;
	  bool is_fine = false;
	  while(!is_fine) {
		   is_fine = true;
		   ind2 = rand()%num_tiles;
		   for(int j = 0; j < num_tiles; j++ ) if(rand_indices[j]==ind2){ is_fine=false; break; }
	  }
	  rand_indices[ind] = ind2;
  }
  sub_num_top[n] = num_sub;
  for (int i = 0; i < num_tiles; ++i) {
    //const Dtype* bottom_data1 = bottom[i]->cpu_data();
    ///const Dtype* bottom_data2 = bottom[i+num_tiles]->cpu_data();
    //const Dtype* top_data = top[i]->cpu_data();
    if( copy_mask[i] == false )
		caffe_copy( copy_size, (Dtype*)bottom[i]->cpu_data() + n*copy_size, (Dtype*)top[i]->cpu_data()+n*copy_size );
	else
		caffe_copy( copy_size, (Dtype*)bottom[num_tiles + rand_indices[i]]->cpu_data()+n*copy_size, (Dtype*)top[i]->cpu_data()+n*copy_size );
  }
  
}//end for n
  delete copy_mask;
  delete rand_indices;
  //offset_concat_axis += bottom_concat_axis;
  //printf("*** %i, %i, %i, %i, %i", bottom.size(), num_concats_, concat_input_size_, concat_axis_, top_concat_axis );
}

#ifdef CPU_ONLY
STUB_GPU(MergeDataLayer);
#endif

INSTANTIATE_CLASS(MergeDataLayer);
REGISTER_LAYER_CLASS(MergeData);

}  // namespace caffe
