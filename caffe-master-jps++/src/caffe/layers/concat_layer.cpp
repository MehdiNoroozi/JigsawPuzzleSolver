#include <vector>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
      << "Either axis or concat_dim should be specified; not both.";
      
  if( concat_param.has_hash_file() )
  {
		
		const string& hash_file = concat_param.hash_file();
		if (Caffe::root_solver()) {
		  LOG(INFO) << "Loading hash file frommmmmmmmmmmmmmmmm: " << hash_file;
		}

		FILE * FID = fopen( hash_file.c_str(), "rb");
		CHECK(FID != NULL)  << "Cannot find hash file";

		size_t t = fread(hashes_sz_, sizeof(int), 2, FID);
		CHECK_EQ(t, 2)  << "Cannot read hash size : " << t;

		LOG(INFO) << "hash size: " << hashes_sz_[0] << " , " << hashes_sz_[1];

		hashes_ = new int[hashes_sz_[0]*hashes_sz_[1]];
		t = fread( hashes_, 4/*sizeof(int)*/, hashes_sz_[0]*hashes_sz_[1], FID );
		CHECK_EQ(t, hashes_sz_[0]*hashes_sz_[1])  << "Cannot read hash : " << t;
		
		for( int i = 0; i < hashes_sz_[0]*hashes_sz_[1]; i++ )  hashes_[i] -= 1;//hashes_[i] =  8-i%9;

		fclose(FID);
		
		if( bottom.size() != hashes_sz_[0] )
		{
			CHECK_EQ( hashes_sz_[1], bottom.size())  << "Number of hashes hase to be the same as the number of input blobs : " << t;
		}
  }
  
  else
  {	  
	  hashes_sz_[0] = 1;
	  hashes_sz_[1] = bottom.size();
	  hashes_ = new int[hashes_sz_[0]*hashes_sz_[1]];
	  for( int i = 0; i < hashes_sz_[0]*hashes_sz_[1]; i++ ) 		hashes_[i] = i%9;//hashes_[i] -= 1;
  }
  temp_hash_len = 0;
  
  hashes_num = hashes_sz_[0];
  hash_len = hashes_sz_[1];

  
}

template <typename Dtype>
void ConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  if (concat_param.has_concat_dim()) {
    concat_axis_ = static_cast<int>(concat_param.concat_dim());
    // Don't allow negative indexing for concat_dim, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(concat_axis_, 0) << "casting concat_dim from uint32 to int32 "
        << "produced negative result; concat_dim must satisfy "
        << "0 <= concat_dim < " << kMaxBlobAxes;
    CHECK_LT(concat_axis_, num_axes) << "concat_dim out of range.";
  } else {
    concat_axis_ = bottom[0]->CanonicalAxisIndex(concat_param.axis());
  }
  // Initialize with the first blob.
  vector<int> top_shape = bottom[0]->shape();
  num_concats_ = bottom[0]->count(0, concat_axis_);
  concat_input_size_ = bottom[0]->count(concat_axis_ + 1);
  int bottom_count_sum = bottom[0]->count();
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num_axes, bottom[i]->num_axes())
        << "All inputs must have the same #axes.";
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) { continue; }
      CHECK_EQ(top_shape[j], bottom[i]->shape(j))
          << "All inputs must have the same shape, except at concat_axis.";
    }
    bottom_count_sum += bottom[i]->count();
    top_shape[concat_axis_] += bottom[i]->shape(concat_axis_);
  }
  top[0]->Reshape(top_shape);
  
  //printf("\n########################  %i", top.size() );
  //getchar();
	if( top.size() > 1 )
	{
		vector<int> top_shape_hash_id = top_shape;
		top_shape_hash_id[0] = top_shape[0];
		top_shape_hash_id[1] = 1;
		top[1]->Reshape(top_shape_hash_id);
	}
	
	if( temp_hash_len != num_concats_ )
	{
		
		//printf("\n^^^^^^^^^^^^^^^^^^^^^ %i, %i ^^^^^^^^^^^^^^^", num_concats_, temp_hash_len );
		temp_hash_len = num_concats_;
		temp_hash_ids = new Dtype[num_concats_];
		for( int i = 0; i < num_concats_; i++ )
			temp_hash_ids[i] = 0;
		if( Caffe::mode() == Caffe::GPU )
		{
			  CUDA_CHECK(cudaMalloc( &d_hashes_, sizeof(int)*hashes_num*hash_len));
			  CUDA_CHECK(cudaMemcpy(d_hashes_, hashes_, sizeof(int)*hashes_num*hash_len ,cudaMemcpyHostToDevice));
			  
			   CUDA_CHECK(cudaMalloc( &d_temp_hash_ids, sizeof(Dtype)*num_concats_) );
			   CUDA_CHECK(cudaMemcpy(d_temp_hash_ids, temp_hash_ids, 
					sizeof(Dtype)*num_concats_ ,cudaMemcpyHostToDevice));
	   }
   }
  
  
	CHECK_EQ(bottom_count_sum, top[0]->count());
	if (bottom.size() == 1) {
		top[0]->ShareData(*bottom[0]);
		top[0]->ShareDiff(*bottom[0]);
	}

}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) { return; }
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  Dtype* top_data_hash_id;
  if( hashes_num > 1 )
  {
	  top_data_hash_id = top[1]->mutable_cpu_data();
	  for (int n = 0; n < num_concats_; ++n)
			top_data_hash_id[n] =  rand() % hashes_sz_[0];

  }
  else
  {
	  top_data_hash_id = temp_hash_ids;
  }

  //int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  
  //printf("\n--- " );
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    //printf(" %i ", bottom_concat_axis );
    for (int n = 0; n < num_concats_; ++n) {
		int * hash_tbl = hashes_ + hashes_sz_[1]*int(top_data_hash_id[n]);
		const int ii = hash_tbl[i];
		caffe_copy(bottom_concat_axis * concat_input_size_,
          bottom_data + n * bottom_concat_axis * concat_input_size_,
          top_data + (n * top_concat_axis + /*offset_concat_axis*/ ii*bottom_concat_axis)
              * concat_input_size_);
    }
    //offset_concat_axis += bottom_concat_axis;
  }
  //printf("*** %i, %i, %i, %i, %i", bottom.size(), num_concats_, concat_input_size_, concat_axis_, top_concat_axis );
}

template <typename Dtype>
void ConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->cpu_diff();
  //int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  
  Dtype* top_data_hash_id;
  Dtype temp = 1;
  if( hashes_num > 1 )
  {
	   top_data_hash_id = top[1]->mutable_cpu_data();
  }
  else
  {
	  top_data_hash_id = temp_hash_ids;
  }

  
  for (int i = 0; i < bottom.size(); ++i) {
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    //printf("\n ^^^ : %i" , propagate_down[i] );
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      for (int n = 0; n < num_concats_; ++n) {
			int * hash_tbl = hashes_ + hashes_sz_[1]*int(top_data_hash_id[n]);
			const int ii = hash_tbl[i];
			caffe_copy(bottom_concat_axis * concat_input_size_, top_diff +
				(n * top_concat_axis + /*offset_concat_axis*/ ii*bottom_concat_axis) * concat_input_size_,
				bottom_diff + n * bottom_concat_axis * concat_input_size_);
      }
    }
    //offset_concat_axis += bottom_concat_axis;
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConcatLayer);
#endif

INSTANTIATE_CLASS(ConcatLayer);
REGISTER_LAYER_CLASS(Concat);

}  // namespace caffe
