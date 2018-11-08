id = 6;
list = dir('imagens-cor-segmentacao');
list = list(3:end);
imagename = list(id).name;

I = imread(sprintf('imagens-cor-segmentacao/%s', imagename));
figure;
imshow(I);

I2 = zeros(size(I,1), size(I,2));

if id == 1
    HSV = rgb2hsv(I);
    for ii = 1:size(HSV,1)
        for jj = 1:size(HSV,2)
            if HSV(ii,jj,1) > 0.13 && HSV(ii,jj,1) < 0.15
                I2(ii,jj,:) = 255;
            else
                I2(ii,jj,:) = 0;
            end
        end
    end
    se = strel('disk', 7);
    I2 = imopen(I2, se);
    se = strel('disk', 11);
    I2 = imdilate(I2, se);

elseif id == 2
    HSV = rgb2hsv(I);
    for ii = 1:size(HSV,1)
        for jj = 1:size(HSV,2)
            if HSV(ii,jj,1) > 0.0 && HSV(ii,jj,1) < 0.04
                I2(ii,jj,:) = 255;
            else
                I2(ii,jj,:) = 0;
            end
        end
    end
    se = strel('disk', 7);
    I2 = imopen(I2, se);
    se = strel('disk', 11);
    I2 = imdilate(I2, se);

elseif id == 3
    HSV = rgb2hsv(I);
    for ii = 1:size(HSV,1)
        for jj = 1:size(HSV,2)
            if HSV(ii,jj,1) > 0.07 && HSV(ii,jj,1) < 0.13
                I2(ii,jj,:) = 255;
            else
                I2(ii,jj,:) = 0;
            end
        end
    end
    se = strel('disk', 9);
    I2 = imopen(I2, se);
    se = strel('disk', 5);
    I2 = imdilate(I2, se);
elseif id == 4
    I2 = im2bw(I, 0.3);
    I2 = ~I2;

elseif id == 5
    HSV = rgb2hsv(I);
    for ii = 1:size(HSV,1)
        for jj = 1:size(HSV,2)
            if HSV(ii,jj,1) > 0.2 && HSV(ii,jj,1) < 0.5
                I2(ii,jj,:) = 0;
            else
                I2(ii,jj,:) = 255;
            end
        end
    end
    se = strel('disk', 7);
    I2 = imopen(I2, se);
    % se = strel('disk', 3);
    % I2 = imdilate(I2, se);

elseif id == 6
    HSV = rgb2hsv(I);
    for ii = 1:size(HSV,1)
        for jj = 1:size(HSV,2)
            if HSV(ii,jj,1) > 0.2 && HSV(ii,jj,1) < 0.5
                I2(ii,jj,:) = 0;
            else
                I2(ii,jj,:) = 255;
            end
        end
    end
    se = strel('disk', 7);
    I2 = imopen(I2, se);
    % se = strel('disk', 3);
    % I2 = imdilate(I2, se);
end

figure;
imshow(I2);

bw = imbinarize(double(I2));
D = bwdist(bw);
DL = watershed(D);
bgm = DL == 0;
fgm = imregionalmax(I2);
se2 = strel(ones(5,5));
fgm2 = imclose(fgm,se2);
fgm3 = imerode(fgm2,se2);
fgm4 = bwareaopen(fgm3,10);
gmag = imgradient(I2);
gmag2 = imimposemin(gmag, bgm | fgm4);

figure;
imshow(gmag2);
L = watershed(gmag2, 4);
L(~bw) = 0;
Lrgb = label2rgb(L);
figure;
imshow(Lrgb);

imwrite(Lrgb, sprintf('output/%d-w.jpg', id));