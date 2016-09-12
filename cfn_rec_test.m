%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Loading network parametrs...
if( exist('load_net') == 0 )
    load_net = 1;
end
if( load_net == 1 )
    
    addpath('./caffe-master-jps/matlab');
    path = './';
    net_model = [path '/deploy_cfn_rec.prototxt'];
    net_weights = ['cfn_rec.caffemodel'];
    net = caffe.Net(net_model, net_weights, 'test');
    load_net = 0;
end

caffe.set_mode_gpu();
crop_size = 225;
mean_value = single(reshape(repmat([104 117 123], 255*255, 1), 255, 255, 3));

db_path  = '/path/to/imagenet/256x256/dataset';
caffe.open_dataset(db_path);
labels  = [];
n = 20;
m = 20;

input = single(zeros(crop_size/3, crop_size/3, 27, m*n));
net.blob_vec(1).reshape([crop_size/3, crop_size/3 27 m*n]);
preds = [];
labels = [];
feats = [];
patch_size = crop_size/3;
labels = [];
preds = [];
cntr = 0;
J = single(zeros(256, 256,3));

while(1)
    k = 1;
    [data,labels_, l] = caffe.read_next_batch(n);

    if( n ~= l )
        break;
    end
    
    for j = 1 : n 
        img = single(data(:,:,:,j));% - mean_value;
         %J(:,:,1) = integralImage(img(:,:,1))./(crop_size*crop_size);
         %J(:,:,2) = integralImage(img(:,:,2))./(crop_size*crop_size);
         %J(:,:,3) = integralImage(img(:,:,3))./(crop_size*crop_size);
            
        for i = 1 : m
            xys = randi(size(img,1)-crop_size,2,1);%([size(img,1) size(img,2)]-crop_size)/2+1;%
            I = img(xys(1):xys(1)+crop_size-1, xys(2):xys(2)+crop_size-1,: );

            for q = 1 : 3
                for p = 1 : 3
                    patch_pos = ((q-1)*3 + p-1)*3+1:((q-1)*3 + p)*3;
                    I_ = I( (p-1)*patch_size+1:p*patch_size, (q-1)*patch_size+1:q*patch_size, : );
                    %mean_val = J(xys(1)+p*patch_size,xys(2)+q*patch_size,:) - ...
                     %   J(xys(1)+(p-1)*patch_size+1,xys(2)+(q-1)*patch_size+1,: );
                    %mean_img = reshape(repmat([mean_val(1) mean_val(2) mean_val(3)],...
                     %                  patch_size*patch_size,1), [patch_size patch_size 3] );
                    I_(:,:,1) = I_(:,:,1) - mean(mean(I_(:,:,1)));
                    I_(:,:,2) = I_(:,:,2) - mean(mean(I_(:,:,2)));
                    I_(:,:,3) = I_(:,:,3) - mean(mean(I_(:,:,3)));
                    input( :,:, patch_pos, (j-1)*m + i ) = I_;
                    here = 1;
                end
            end
        end
    end
    
    input_data = {input};
    scores = net.forward(input_data);
    preds_ = reshape(scores{1}, 1000, m, n) ;
    preds_ = mean(preds_, 2);
    [v,i] = max(preds_,[],1);
    preds = [preds; i(:)-1];
    labels = [labels; labels_'];
    %disp([num2str(i-1) ' , ' num2str(labels_)]);
    here = 1;
    cntr = cntr + n;
    if( mod(cntr,100)==0)
        disp([num2str(cntr) ' : ' num2str(length(find(  preds==labels  ))*100/length(labels))]);
    end
end
