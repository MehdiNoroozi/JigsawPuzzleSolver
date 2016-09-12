%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Loading network parametrs...
if( exist('load_net') == 0 )
    load_net = 1;
end
if( load_net == 1 )
    
    addpath('./caffe-master-jps/matlab');
    path = './';
    net_model = [path '/deploy_cfn_jps.prototxt'];
    net_weights = ['cfn_jps.caffemodel'];
    net = caffe.Net(net_model, net_weights, 'test');
    load_net = 0;
end

%%loading permutations 
FID = fopen('permutations_100_max.bin', 'rb');
pset_sz = fread(FID, [1 2],'int32');
pset = fread(FID, pset_sz,'int32');
pset = reshape(pset, pset_sz(2), pset_sz(1))';
fclose(FID);
%%
caffe.set_mode_gpu();
crop_sz = 255;
patch_sz = 64;
cell_sz = crop_sz/3; 
mean_value = single(reshape(repmat([104 117 123], crop_sz*crop_sz, 1), crop_sz, crop_sz, 3));

db_path  = '/path/to/jps/dataset/'
caffe.open_dataset(db_path);
labels  = [];

input = single(zeros(patch_sz, patch_sz, 27, 1));
input_ = single(zeros(patch_sz, patch_sz, 27, 1));
net.blob_vec(1).reshape([patch_sz, patch_sz, 27, 1]);
labels = [];
preds = [];
cntr = 0;

while(1)
    k = 1;
    [data,labels_, l] = caffe.read_next_batch(1);

    if( l ~= 1 )
        break;
    end
       
    img = single(data);
    if( crop_sz == size(img,1) )
        I = img;
    else
        xys = randi(size(img,1)-crop_size,2,1);
        I = img(xys(1):xys(1)+crop_size-1, xys(2):xys(2)+crop_size-1,: );
    end

    for q = 1 : 3
        for p = 1 : 3
            patch_pos = ((q-1)*3 + p-1)*3+1:((q-1)*3 + p)*3;
            xys_ = randi(cell_sz-patch_sz,2,1);
            I_ = I( (p-1)*cell_sz+1:p*cell_sz, (q-1)*cell_sz+1:q*cell_sz, : );
            I_ = I_(xys_(1):xys_(1)+patch_sz-1, xys_(2):xys_(2)+patch_sz-1,: );
            I_(:,:,1) = I_(:,:,1) - mean(mean(I_(:,:,1)));
            I_(:,:,2) = I_(:,:,2) - mean(mean(I_(:,:,2)));
            I_(:,:,3) = I_(:,:,3) - mean(mean(I_(:,:,3)));
            input_( :,:, patch_pos, 1 ) = I_;
            here = 1;
        end
    end
    
    perm_id = randi(pset_sz(1),1);

    perm = pset(perm_id, :);
    for i = 1 : 9
        input(:,:,3*(i-1)+1:3*i) = input_(:,:,3*(perm(i)-1)+1:3*(perm(i)-1)+3);
    end
    
    input_data = {input};
    scores = net.forward(input_data);
    scores = scores{1};
    [v,i] = max(scores);
    disp(['perm_id: ' num2str(perm_id), ' , predicted: ' num2str(i) ' , confidency: ' num2str(int32(v*100))]);
end
