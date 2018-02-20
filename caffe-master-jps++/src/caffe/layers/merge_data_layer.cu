#include <vector>

#include "caffe/layers/merge_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void MergeDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

 int num_tiles = top.size() - 1;


  bool * copy_mask = new bool[num_tiles];
  int * rand_indices = new int[num_tiles];
  int copy_size = bottom[0]->count(1);
  //printf("\n **************  %i, %i, %i *************", copy_size, bottom[0]->num(), num_tiles  );
  Dtype * sub_num_top = top[top.size()-1]->mutable_gpu_data();
  for( int n = 0; n < bottom[0]->num(); n++ )
  {
  	Dtype num_sub = rand()%max_substitute_ + min_substitute_;
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
	CUDA_CHECK(cudaMemcpy(sub_num_top+n, &num_sub, sizeof(Dtype) ,cudaMemcpyHostToDevice)); //sub_num_top[n] = num_sub;
 	 for (int i = 0; i < num_tiles; ++i) {
	 if( copy_mask[i] == false )
                	CUDA_CHECK(cudaMemcpy( (Dtype*)top[i]->gpu_data()+n*copy_size, (Dtype*)bottom[i]->gpu_data() + n*copy_size,
				sizeof(Dtype)*copy_size, cudaMemcpyDeviceToDevice ));
        	else
                        CUDA_CHECK(cudaMemcpy( (Dtype*)top[i]->gpu_data()+n*copy_size, (Dtype*)bottom[num_tiles + rand_indices[i]]->gpu_data() + n*copy_size, 
                                sizeof(Dtype)*copy_size, cudaMemcpyDeviceToDevice ));
  	}
        
  }
  delete copy_mask;
  delete rand_indices;
}

INSTANTIATE_LAYER_GPU_FUNCS(MergeDataLayer);

}  // namespace caffe
