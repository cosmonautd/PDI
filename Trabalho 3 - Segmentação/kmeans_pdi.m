id = 6;
list = dir('imagens-cor-segmentacao');
list = list(3:end);
imagename = list(id).name;

I = imread(sprintf('imagens-cor-segmentacao/%s', imagename));
I = medfilt3(I);
figure;
imshow(I);

K = [3 3 3 2 2 4];

[lb,center] = adaptcluster_kmeans(I);

S2 = mat2gray(lb);

figure;
imshow(S2);

center

imwrite(S2, sprintf('output/%d-km.jpg', id));