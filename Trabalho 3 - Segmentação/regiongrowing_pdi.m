seeds = zeros(6,5,2);
seeds(1,:,:) = [180 250; 80 172; 300 200; 190 115; 96 64];
seeds(2,:,:) = [150 150; 300 125; 70 80; 192 78; 50 20];
seeds(3,:,:) = [360 175; 165 240; 110 225; 145 205; 388 148];
seeds(4,:,:) = [217 150; 300 212; 365 244; 262 223; 377 182];
seeds(5,:,:) = [41 237; 104 204; 213 242; 317 241; 407 238];
seeds(6,:,:) = [273 77; 276 136; 310 169; 334 161; 301 123];

tresholds = [35 40 15 35 18 12];

id = 6;
list = dir('imagens-cor-segmentacao');
list = list(3:end);
imagename = list(id).name;

I = imread(sprintf('imagens-cor-segmentacao/%s', imagename));
I = rgb2gray(I);
I = medfilt2(I, [3 3]);
figure;
imshow(I);

S = regiongrowing(double(I), seeds(id,1,2), seeds(id,1,1), tresholds(id));
for ii = 2:5
    S = S + regiongrowing(double(I), seeds(id,ii,2), seeds(id,ii,1), tresholds(id));
end

figure;
imshow(double(S));
imwrite(S, sprintf('output/%d-rg.jpg', id));