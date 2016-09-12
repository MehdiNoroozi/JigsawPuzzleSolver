

//#include <gflags/gflags.h>
//#include <glog/logging.h>
//#include <google/protobuf/text_format.h>
//#include <leveldb/db.h>
//#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>
//#include <cv.h>
#include <cv.hpp>
#include <highgui.h>

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/db.hpp"
#include "boost/scoped_ptr.hpp"
#include <dirent.h>

#define PI 3.14159


using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

using std::max;
using std::pair;
using boost::scoped_ptr;
        

//DEFINE_string(backend, "lmdb", "The backend for storing the result");

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

// lmdb
MDB_env *mdb_env = NULL;
MDB_dbi mdb_dbi;
MDB_val mdb_key, mdb_data;
MDB_txn *mdb_txn;
const int kMaxKeyLength = 20;
char key_cstr[kMaxKeyLength];
string value;
int item_id = 0;
int hn_cnt = 0;
BlobProto sum_blob;
int patch_stride;


void AddtoDb(IplImage * img, int label )
{	
	//printf("%.2f, %.2f, %.2f", s.val[0], min_val, max_val );
	//getchar();
			
			
	//ShowStack(stack, l, 64, 11);
	Datum  datum;
	//Add to Datum...
	datum.set_channels(img->nChannels);
	datum.set_height(img->height);
	datum.set_width(img->width);
	datum.clear_data();
	datum.clear_float_data();
	datum.set_encoded(false);
	int datum_channels = datum.channels();
	int datum_height = datum.height();
	int datum_width = datum.width();
	int datum_size = datum_channels * datum_height * datum_width;
	std::string buffer(datum_size, ' ');
	char * data = &(buffer[0]);
	
	int datum_offset = 0;



	const uchar* ptr1 = (unsigned char *)img->imageData;
	int k = 0;
	int channel_stride = datum_width*datum_height;
	for (int h = 0; h < datum_height; ++h) 
	{
		for (int w = 0; w < datum_width; ++w) 
		{
			for( int c = 0; c < img->nChannels; c++ )
			{
				buffer[k + c*channel_stride] = static_cast<char>(ptr1[h*img->widthStep + img->nChannels*w + c]);
				sum_blob.set_data(	k + c*channel_stride,
									sum_blob.data(k + c*channel_stride) +
									(uint8_t)ptr1[h*img->widthStep + img->nChannels*w + c] );
			}
			k = k + 1;
		}
	}
	
	datum.set_data(buffer);
	datum.set_label(label);
	

	//Add to database			
	snprintf(key_cstr, kMaxKeyLength, "%08d", item_id);
	datum.SerializeToString(&value);
	string keystr(key_cstr);
	mdb_data.mv_size = value.size();
	mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
	mdb_key.mv_size = keystr.size();

	mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
	mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);
	if (item_id % 10000 == 0) 
	{
		time_t t = time(0);   // get time now
		struct tm * now = localtime( & t );
    
		mdb_txn_commit(mdb_txn);
		mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn);
	
		printf("\n%02i : %02i : %02i *** %07i  *** %07i ***", now->tm_hour, now->tm_min, now->tm_sec, item_id, hn_cnt );
	}
}



IplImage * DatumToIplImage(Datum * datum )
{

	const std::string data = datum->data();
	int size_in_datum = std::max<int>(datum->data().size(),
	datum->float_data_size());
	IplImage * img;
	if (data.size() != 0) 
	{
		int k = 0;
		
		int width  = datum->width();
		int height = datum->height();
		int channel_stride = width*height;
		
		img = cvCreateImage(cvSize(datum->width(), datum->height()), 8, 3);
		for( int j = 0; j < height; j++ )
		{
			for( int i = 0; i < width; i++ )
			{
				img->imageData[j*img->widthStep + 3*i + 0] = (char)data[k]; 
				img->imageData[j*img->widthStep + 3*i + 1] = (char)data[k+channel_stride]; 
				img->imageData[j*img->widthStep + 3*i + 2] = (char)data[k+2*channel_stride];
				k = k + 1;
		  }
		}
	}
	else 
	{
		CHECK_EQ(datum->float_data_size(), size_in_datum);
	}
	return img;	
}

int main(int argc, char** argv) 
{	
	::google::InitGoogleLogging(argv[0]);
	bool show_flg = false;
	const string db_backend = "lmdb";// FLAGS_backend;
	printf("\n*******************************************************");
	printf("\nGenerates dataset to be used in training of jigsaw puzzle solver" );
	printf("\n*******************************************************");
	printf("\nInput DB Path  : %s", argv[1]);
	printf("\nOutput DB Path  : %s", argv[2]);
	printf("\n*******************************************************");

	int depth = 3;
	int rows = 255;
	int cols = 255;
	int count = 0;
  	sum_blob.set_num(1);
	sum_blob.set_channels(depth);
	sum_blob.set_height(rows);
	sum_blob.set_width(cols);
	const int data_size = rows * cols * depth;
	for (int i = 0; i < rows*cols*depth; ++i)
	{
		sum_blob.add_data(0.);
	}
  				
	LOG(INFO) << "Starting Iteration";
  
  
  	mkdir(argv[2], 0744);
  	mdb_env_create(&mdb_env);
	mdb_env_set_mapsize(mdb_env, 1099511627776);
    mdb_env_open(mdb_env, argv[2], 0, 0664);
    mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn);
    mdb_open(mdb_txn, NULL, 0, &mdb_dbi);
	
	int db_id = 0;
	
	count = 0;
    
    IplImage * temp_img_c = cvCreateImage( cvSize(255,255), 8, 3 );
	item_id = 0;

  	
	scoped_ptr<db::DB> db(db::GetDB("lmdb"));
	db->Open(argv[1], db::READ);
	scoped_ptr<db::Cursor> cursor( db->NewCursor());
	int cntr = 0;
	while (cursor->valid()) 
	{
		Datum datum;
		datum.ParseFromString(cursor->value());
		DecodeDatumNative(&datum);

		IplImage * img = DatumToIplImage(&datum);
		CvPoint corner = cvPoint( (img->width - temp_img_c->width)/2,
							   (img->height - temp_img_c->height)/2 );
		CvRect rct = cvRect( corner.x, corner.y, temp_img_c->width, temp_img_c->height );
						  cvSetImageROI(img, rct );
		cvSetImageROI(img, rct );
		cvCopy( img, temp_img_c );
		AddtoDb(temp_img_c, datum.label());
		item_id++;

		cvReleaseImage( &img );
		
		cursor->Next();
	}
		
	mdb_txn_commit(mdb_txn);
	mdb_close(mdb_env, mdb_dbi);
	mdb_env_close(mdb_env);

	char str_temp[200];
    sprintf( str_temp, "%s/mean.binaryproto", argv[2] );
	for (int i = 0; i < sum_blob.data_size(); ++i)
	{
		sum_blob.set_data(i, sum_blob.data(i) / item_id);
	}
	// Write to disk
	WriteProtoToBinaryFile(sum_blob, str_temp);
	
	printf("\n");

	return 0;
}
