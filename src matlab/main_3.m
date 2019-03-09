%2017-01-05
% ACF function comparation

close all,clc,clear all;

%imgA = imread('D:\\home\\programming\\vc\\old\\bmp\\3.bmp');
imgA = imread('..\\input\\3.bmp');
%imgA = imread('D:\\home\\programming\\vc\\new\\5_Alanis Software\\1_SplitPage\\example2\\resized\\00000110_resized.TIF');

imgA = rgb2gray(imgA); %color->gray
figure; imshow(imgA);
title('Original image');

%PSD calculation
imgA = double(imgA);
img_fft = fft2(imgA);           %spectrum 
img_fft(1,1) = 0;               %removing of constant component
%imgB = img_fft.*conj(img_fft); %Power spectrum density
%imgB = sqrt(imgB);             %sqrt(Power spectrum density)
imgB = abs(img_fft);            %sqrt(Power spectrum density)

imgC = fftshift(255*(imgB -min(min(imgB))) /(max(max(imgB)) - min(min(imgB))));
imgClog = log(1+imgB);
imgClog_norm = fftshift(255*(imgClog -min(min(imgClog))) /(max(max(imgClog)) - min(min(imgClog))));

%ACF calculation
imgD = ifft2(imgB);
imgD(1,1) = 0;
imgE = 255*(imgD -min(min(imgD))) /(max(max(imgD)) - min(min(imgD)));
imgE = fftshift(imgE);


%output
imwrite(uint8(imgC),'..\\output\\PSD_m.jpg');
imwrite(uint8(imgE),'..\\output\\ACF_m.jpg');
imwrite(uint8(imgClog_norm),'..\\output\\PSD_log_m.jpg');
figure, imshow(uint8(imgC));
title('Power spectrum density');
figure, imshow(uint8(imgClog_norm));
title('Power spectrum density Log');
figure, imshow(uint8(imgE));
title('Auto correlation function');