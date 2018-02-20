

db = lmdb.DB('/home/mehdi/Desktop/SyntheticShapes/C++/Caffe_training/test', 'RDONLY', true, 'NOLOCK', true);

cursor = db.cursor('RDONLY', true);
len = 0;
tic;
while cursor.next()
  
  key = cursor.key;
  value = cursor.value;
  I = uint8(reshape(value(17:end), [230 230 3]));
  a = I(:,:,3);
  I(:,:,3) = I(:,:,1);
  I(:,:,1) = a;
  disp(key);
  figure(1), imshow(I);
  pause(1);
  len = len + 1;
  if(mod(len,1000) == 0 )
      toc;
      tic;
  end
end