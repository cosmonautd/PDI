id = 6;
list = dir('imagens-cor-segmentacao');
list = list(3:end);
imagename = list(id).name;

I = imread(sprintf('imagens-cor-segmentacao/%s', imagename));
I = rgb2gray(I);
I = medfilt2(I, [3 3]);
figure;
imshow(I);

T = [2 2 1 1 1 1];

thresh = multithresh(I,T(id));

S = imquantize(I, thresh);
RGB = mat2gray(S);
figure;
imshow(RGB);

imwrite(RGB, sprintf('output/%d-o.jpg', id));