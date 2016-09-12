
addpath('./liblinear/matlab');
FID = fopen('/home/mehdi/Desktop/FileZilaDownload/ref_patches.bin');
count = fread(FID, 1, 'int32');
patch_sz = fread(FID, 1, 'int32');
chn = fread(FID, 1, 'int32');
patch_stride = patch_sz*patch_sz*chn;
patches = zeros(patch_stride/4 + 1, count );
patches(end,:) = 1;
tls = zeros(count,2);
for  i = 1 : count
    
    tls(i,:) = fread(FID, 2, 'int32');
    
    img = reshape(fread(FID, patch_stride, 'uint8'),[patch_sz patch_sz chn]);
    img = img(1:2:end, 1:2:end,:);
    patches(1:end-1,i) = img(:);
    %figure(1), imshow(img,[]);
    %pause(.001);
end

Np = 1;
Nn = double(int32(.01*count));
N = Nn + Np;

norm = 2;
c = 10;

if norm==2
   opt1 = ['-q -B 1 -c ' num2str(c) ' -s 1 -w-1 ' num2str((1./Nn)) ' -w1 ' num2str(1/Np)];
else
   opt1 = ['-q -B 1 -c ' num2str(c) ' -s 6 -w-1 ' num2str((Np/N)) ' -w1 ' num2str(Nn/N)];
end
  
phi = sparse(count,count);
clc;
tic;
sum_pos = 0;
for i = 1 : count
    neg_inds = randi([1 count],[1 Nn]);
    posneg_train_data = patches(1:end-1,i);
    posneg_train_data = [posneg_train_data patches(1:end-1,neg_inds)];
    posneg_train_label = ones(1,Nn+Np)*-1;
    posneg_train_label(1) = 1;
    models = train(posneg_train_label',sparse(posneg_train_data),...
        opt1,'col');
    s = models.w*patches;
    inds = find(s>0);
    sum_pos = sum_pos + length(inds);
    [v,ii] = sort(s(inds), 'descend');
    iii = zeros(1,length(ii));
    iii(ii) = [1:length(ii)];
    phi(i,inds) = iii;
    
    if( mod(i,100) == 0 )
        t = toc;
        disp([num2str(i) ':' num2str(t) ' , ' num2str(sum_pos/i)]);
        tic;
    end
end
save('/var/tmp/phi.mat', 'phi');
save('/var/tmp/tls.mat', 'tls');