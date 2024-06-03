%Jessi Singh
%BIOENG 
%Systems Neuroscience
%Jan 15, 2024

%% Step 1: 
%a) Create the visual input for the Mach Band Illusion
input_matrix = zeros(64, 128); % Initialize a 64 by 128 matrix

%Linearly increase from 10 to 32 for the first 32 columns
input_matrix(:, 1:32) = repmat(linspace(10, 32, 32), 64, 1);

%Linearly increase from 32 to 64 for the next 32 columns
input_matrix(:, 33:64) = repmat(linspace(32, 64, 32), 64, 1);

%Set the rest of the columns to 75
input_matrix(:, 65:end) = 75;


%% step 1 alternate
% Create portions of Mach Band
part1 = ones(64, 32) * 10;
part2 = repmat(11:75, 64, 1);
part3 = ones(64, 31) * 75;

% Concatenate to create the Mach Band matrix
input_matrix = [part1, part2, part3];

% Display the Mach Band image
figure('Color', 'w');
imagesc(input_matrix);
colormap gray;
title('Mach Band Illusion', 'FontSize', 20, 'FontWeight', 'normal', 'Color', 'b');
xlabel('Horizontal Image Position');
ylabel('Vertical Image Position');



%%
% Display the input image
figure('Color', 'w');
colormap gray;
imagesc(input_matrix);
title('Mach Band Illusion for the Visual Input', 'FontSize',20,'FontWeight','normal','Color','b');
xlabel('Horizontal Position');
ylabel('Vertical Position');
% Part 1: Mach Band Illusion
% 
% % a) Create the visual input for the Mach Band Illusion
% %step 1 attempt 2
% input_matrix = zeros(64, 128); %because 64 by 128
% input_matrix(:, 1:32) = repmat(10:32, 32, 1);
% input_matrix(:, 33:64) = 64+.1;
% input_matrix(:, 64:end) = 75;
% % 
% % step 1 attempt 1 
% imageMatrix(:, 1:32) = 10;
% imageMatrix(:, 33+75) = 75;
% 
% figure;
% imagesc(input_matrix);
% colormap gray;
% title('Mach Band Illusion for the Visual Input');
%% step 1 attept 3
% 
% % Problem 1a
% image_matrix = ones(64, 128) * 75;
% image_matrix(:, 1:32) = repmat(linspace(10, 32, 32), 64, 1); % Linearly increase from 10 to 32
% image_matrix(:, 33:64) = repmat(linspace(32, 64, 32), 64, 1); % Linearly increase from 32 to 64
% imagesc(image_matrix);
% 
% figure;
% imagesc(input_matrix);
% colormap gray;
% title('Mach Band Illusion for the Visual Input');


%% attempt 1 
% b) Plot brightness as a function of horizontal position
slice = input_matrix(32, :);
figure('Color', 'w');
plot(slice);

xlabel('Horizontal Position');
ylabel('Brightness');
title('Brightness Profile along Horizontal Slice', 'FontSize',20, 'FontWeight','bold', 'Color','b');
% Set y-axis lower limit to 0
ylim([0, 80]);
% %% attempt 2 - loosk the same as one lol
% % b) Plot brightness as a function of horizontal position
% brightness_slice = mean(input_matrix, 1); % Take the mean along the rows
% figure;
% plot(brightness_slice, 'LineWidth', 2);
% title('Brightness as a Function of Horizontal Position', 'FontSize',20,'FontWeight','normal','Color','b');
% xlabel('Horizontal Position');
% ylabel('Brightness');
% colormap gray;


%% attempt 3 
% c) Create the receptive field of a retinal ganglion cell
% c) Create the receptive field of a retinal ganglion cell
[X, Y] = meshgrid(-2:2);

% Create modified circular Gaussian functions

% Calculate the normalization factor
norm_factor = 1 / (2 * pi * std_x * std_y);

% Calculate the exponential part of the Gaussian
;
Egaus = (2.*pi.*SF.*X) * exp(-(X.^2 + Y.^2) / (2 * 2^2));
Igaus =  (2.*pi.*SF.*X)* exp(-(X.^2 + Y.^2) / (2 * 6^2)); % Invert the Gaussian to make the surround dark

S = 500;
IE = 6;
recf = S * (Egaus - IE * Igaus);

% Display the receptive field
figure;
imagesc(recf);
colormap gray; %hot,jet,pink
% Set the color axis limits to be close to -6 and 4


title('Receptive Field of a Retinal Ganglion Cell', 'FontSize', 20, 'Color','b');
xlabel('horozontal posistion')
ylabel('vertical posistion')

%%
% d) Convolve receptive field and input image
input_matrix = [part1, part2, part3];
recf = S * (Egaus - IE * Igaus);
conv_res = conv2(input_matrix, recf, 'valid');

figure;
imagesc(conv_res);
colormap gray;
title('Output after Convolution', 'FontSize', 20, 'Color','b');

%% attempt d part II 
% d) Convolve your receptive field and input image
convolved_output = conv2(input_matrix, recf, 'valid');

% Display the output of the convolution
figure;
imagesc(convolved_output);
colormap gray;
title('Output of Convolution with Receptive Field', 'FontSize', 14);
xlabel('Horizontal Position');
ylabel('Vertical Position');

% Plot a horizontal slice through the output matrix
figure;
slice_position = size(convolved_output, 1) / 2;
plot(convolved_output(slice_position, :), 'LineWidth', 2);
title('Horizontal Slice through Convolution Output', 'FontSize', 14);
xlabel('Horizontal Position');
ylabel('Brightness');


%% Step 2 

% Part 2: Gabor Function - wrong 
% 
% % a) Define values
% OR = pi/2; % Orientation in radians
% % SF = 0.05; % Spatial frequency
% 
% % b) Create matrices
% [x, y] = meshgrid(-20:20);
% 
% % Skew and rotate the Gaussian
% X = x * cos(OR) + y * sin(OR);
% Y = -x * sin(OR) + y * cos(OR);
% 
% % c) Create Gabor function
% std_x = 7;
% std_y = 17;
% 
% % 2D Gaussian
% % gaussian = exp(-0.5 * ((X/std_x).^2 + (Y/std_y).^2));
% gaussian = exp(-((X.^2 ./ (2.*std_x.^2)) + (Y.^2 ./ (2.*std_y.^2))));
% 
% % Modulate the Gaussian with a sine wave
% A = sin(2.*pi.*SF.*X) ./ (2.*pi.*std_x.*std_y); % (2*pi*SF*X.*Y);
% 
% % Gabor function
% gabor_function = A .* gaussian;
% 
% figure; imagesc(gabor_function); colormap gray; axis square;
% title ('Gabor Function with OR = pi/2 and SF = .05')
% xlabel('Horizontal Posisition')
% ylabel('Vertical Posistion')
% % temp plot out 
% %  Plot the Gabor function
% % surf(gabor_function);
% % title('Gabor Function');

%% step 2 new new new new 

%% Step 1c: Create the receptive field of a retinal ganglion cell - wrong

% Create matrices
[X, Y] = meshgrid(-2:2);

% Convert to polar coordinates
[Theta, Rho] = cart2pol(X, Y);

% Create modified circular Gaussian functions
Egaus = exp(-(Rho.^2) / (2 * 2^2)); % Use element-wise exponentiation
Igaus = 1 - exp(-(Rho.^2) / (2 * 6^2)); % Use element-wise exponentiation, and invert the Gaussian to make the surround dark

S = 500;
IE = 6;
recf = S * (Egaus - IE * Igaus);

% Display the receptive field with colormap gray and colorbar
figure;
imagesc(recf);
colormap gray;
colorbar;

title('Receptive Field of a Retinal Ganglion Cell', 'FontSize', 20, 'Color', 'b');
xlabel('Horizontal Position');
ylabel('Vertical Position');

% d) Convolve receptive field and input image
input_matrix = [part1, part2, part3];
recf = S * (Egaus - IE * Igaus);
conv_res = conv2(input_matrix, recf, 'valid');

figure;
imagesc(conv_res);
colormap gray;
title('Output after Convolution', 'FontSize', 20, 'Color','b');

%% Manipulate the OR pi/2 and .1 

% Part 2: Gabor Function

% a) Define values
OR = pi/4 % Orientation in radians
SF = 0.05; % Spatial frequency

% b) Create matrices
[x, y] = meshgrid(-20:20);

% Skew and rotate the Gaussian
X = x * cos(OR) + y * sin(OR);
Y = -x * sin(OR) + y * cos(OR);

% c) Create Gabor function
std_x = 7;
std_y = 17;

% 2D Gaussian
% gaussian = exp(-0.5 * ((X/std_x).^2 + (Y/std_y).^2));
gaussian = exp(-((X.^2 ./ (2.*std_x.^2)) + (Y.^2 ./ (2.*std_y.^2))));

% Modulate the Gaussian with a sine wave
A = sin(2.*pi.*SF.*X) ./ (2.*pi.*std_x.*std_y); % (2*pi*SF*X.*Y);

% Gabor function
gabor_function = A .* gaussian;

figure; imagesc(gabor_function); colormap gray; axis square;
title ('Gabor Function with OR = pi/4 and SF = .05')
xlabel('Horizontal Posisition')
ylabel('Vertical Posistion')
% temp plot out 
%  Plot the Gabor function
% surf(gabor_function);
% title('Gabor Function');
%%
%%% Part 2: Gabor Function

% a) Define values
OR = pi/2; % Orientation in radians
SF = 0.1; % Spatial frequency

% b) Create matrices
[x, y] = meshgrid(-20:20);

% Skew and rotate the Gaussian
X = x * cos(OR) + y * sin(OR);
Y = -x * sin(OR) + y * cos(OR);

% c) Create Gabor function
std_x = 7;
std_y = 17;

% 2D Gaussian
% gaussian = exp(-0.5 * ((X/std_x).^2 + (Y/std_y).^2));

gaussian = exp(-((X.^2 ./ (2.*std_x.^2)) + (Y.^2 ./ (2.*std_y.^2))));

% Modulate the Gaussian with a sine wave
A = sin(2.*pi.*SF.*X) ./ (2.*pi.*std_x.*std_y); % (2*pi*SF*X.*Y);

% Gabor function
gabor_function = A .* gaussian;

figure; imagesc(gabor_function); colormap gray; axis square;
title ('Gabor Function with OR = pi/2 and SF = .1')
xlabel('Horizontal Posisition')
ylabel('Vertical Posistion')
% temp plot out 
%  Plot the Gabor function
% surf(gabor_function);
% title('Gabor Function');
%% Play with OR (try pi/2 or pi/4) and SF (try .05 or .1) 

%%% Part 2: Gabor Function

% a) Define values
OR = pi/2; % Orientation in radians which is pi/2
SF = 0.05; % Spatial frequency

% b) Create matrices
[x, y] = meshgrid(-20:20);

% Skew and rotate the Gaussian
X = x * cos(OR) + y * sin(OR);
Y = -x * sin(OR) + y * cos(OR);

% c) Create Gabor function
std_x = 7;
std_y = 17;

% 2D Gaussian
% gaussian = exp(-0.5 * ((X/std_x).^2 + (Y/std_y).^2));
gaussian = exp(-((X.^2 ./ (2.*std_x.^2)) + (Y.^2 ./ (2.*std_y.^2))));

% Modulate the Gaussian with a sine wave
A = sin(2.*pi.*SF.*X) ./ (2.*pi.*std_x.*std_y); % (2*pi*SF*X.*Y);

% Gabor function
gabor_function = A .* gaussian;

figure; imagesc(gabor_function); colormap gray; axis square;
title ('Gabor Function with OR = pi/4 and SF = .1')
xlabel('Horizontal Posisition')
ylabel('Vertical Posistion')
% temp plot out 
%  Plot the Gabor function
% surf(gabor_function);
% title('Gabor Function');

% imagesc(gabor_function)
% still step 2 
% e) Load 'rose.jpg' and convolve with Gabor function
% Load 'rose.jpg' and convert to grayscale
% rose_image = imread('C:\Users\jaspreetsingh\MATLAB-Drive\Systems Neuroscience\rose.jpg'); 
rose_image = imread('rose.jpg');
% rose=im2double(imread("rose.jpg"))
gabor_function = A .* gaussian;
gray_rose = double(rose_image);
convolved_image = conv2(gray_rose, gabor_function, 'same');

% Plot original image
figure;
sgtitle('Rose and Convolved Rose with OR = pi/2 and SF =.05')
subplot(1, 2, 1);
imagesc(gray_rose);
colormap gray;
title('Original Image');
xlabel('Horizontal Position (pixel)');
ylabel('Vertical Position(pixel)');

% Plot result of convolution
% figure;
subplot(1, 2, 2);
imagesc(convolved_image);
colormap gray;
title ('Convolved Rose')
xlabel('Horizontal Position (pixel)');
ylabel('Vertical Position(pixel)');
% Assuming convolved_image is your matrix
min_value = min(convolved_image(:));
max_value = max(convolved_image(:));

%%
%% Play with OR (try pi/2 or pi/4) and SF (try .05 or .1) 

%%% Part 2: Gabor Function

% a) Define values
OR = pi/2; % Orientation in radians which is pi/2
SF = 0.1; % Spatial frequency

% b) Create matrices
[x, y] = meshgrid(-20:20);

% Skew and rotate the Gaussian
X = x * cos(OR) + y * sin(OR);
Y = -x * sin(OR) + y * cos(OR);

% c) Create Gabor function
std_x = 7;
std_y = 17;

% 2D Gaussian
% gaussian = exp(-0.5 * ((X/std_x).^2 + (Y/std_y).^2));
gaussian = exp(-((X.^2 ./ (2.*std_x.^2)) + (Y.^2 ./ (2.*std_y.^2))));

% Modulate the Gaussian with a sine wave
A = sin(2.*pi.*SF.*X) ./ (2.*pi.*std_x.*std_y); % (2*pi*SF*X.*Y);

% Gabor function
gabor_function = A .* gaussian;

figure; imagesc(gabor_function); colormap gray; axis square;
title ('Gabor Function with OR = pi/4 and SF = .1')
xlabel('Horizontal Posisition')
ylabel('Vertical Posistion')
% temp plot out 
%  Plot the Gabor function
% surf(gabor_function);
% title('Gabor Function');

% imagesc(gabor_function)
% still step 2 
% e) Load 'rose.jpg' and convolve with Gabor function
% Load 'rose.jpg' and convert to grayscale
% rose_image = imread('C:\Users\jaspreetsingh\MATLAB-Drive\Systems Neuroscience\rose.jpg'); 
rose_image = imread('rose.jpg');
% rose=im2double(imread("rose.jpg"))
gabor_function = A .* gaussian;
gray_rose = double(rose_image);
convolved_image = conv2(gray_rose, gabor_function, 'same');

% Plot original image
figure;
sgtitle('Rose and Convolved Rose 2 with OR = pi/2 and SF =.1')
subplot(1, 2, 1);
imagesc(gray_rose);
colormap gray;
title('Original Image');
xlabel('Horizontal Position (pixel)');
ylabel('Vertical Position(pixel)');

% Plot result of convolution
% figure;
subplot(1, 2, 2);
imagesc(convolved_image);
colormap gray;
title ('Convolved Rose')
xlabel('Horizontal Position (pixel)');
ylabel('Vertical Position(pixel)');
% Assuming convolved_image is your matrix
min_value = min(convolved_image(:));
max_value = max(convolved_image(:));

%%
%% Play with OR (try pi/2 or pi/4) and SF (try .05 or .1) 

%%% Part 2: Gabor Function

% a) Define values
OR = pi/4; % Orientation in radians which is pi/2
SF = 0.05; % Spatial frequency

% b) Create matrices
[x, y] = meshgrid(-20:20);

% Skew and rotate the Gaussian
X = x * cos(OR) + y * sin(OR);
Y = -x * sin(OR) + y * cos(OR);

% c) Create Gabor function
std_x = 7;
std_y = 17;

% 2D Gaussian
% gaussian = exp(-0.5 * ((X/std_x).^2 + (Y/std_y).^2));
gaussian = exp(-((X.^2 ./ (2.*std_x.^2)) + (Y.^2 ./ (2.*std_y.^2))));

% Modulate the Gaussian with a sine wave
A = sin(2.*pi.*SF.*X) ./ (2.*pi.*std_x.*std_y); % (2*pi*SF*X.*Y);

% Gabor function
gabor_function = A .* gaussian;

figure; imagesc(gabor_function); colormap gray; axis square;
title ('Gabor Function with OR = pi/4 and SF = .1')
xlabel('Horizontal Posisition')
ylabel('Vertical Posistion')
% temp plot out 
%  Plot the Gabor function
% surf(gabor_function);
% title('Gabor Function');

% imagesc(gabor_function)
% still step 2 
% e) Load 'rose.jpg' and convolve with Gabor function
% Load 'rose.jpg' and convert to grayscale
% rose_image = imread('C:\Users\jaspreetsingh\MATLAB-Drive\Systems Neuroscience\rose.jpg'); 
rose_image = imread('rose.jpg');
% rose=im2double(imread("rose.jpg"))
gabor_function = A .* gaussian;
gray_rose = double(rose_image);
convolved_image = conv2(gray_rose, gabor_function, 'same');

% Plot original image
figure;
sgtitle('Rose and Convolved Rose 3 with OR = pi/4 and SF =.05')
subplot(1, 2, 1);
imagesc(gray_rose);
colormap gray;
title('Original Image');
xlabel('Horizontal Position (pixel)');
ylabel('Vertical Position(pixel)');

% Plot result of convolution
% figure;
subplot(1, 2, 2);
imagesc(convolved_image);
colormap gray;
title ('Convolved Rose')
xlabel('Horizontal Position (pixel)');
ylabel('Vertical Position(pixel)');
% Assuming convolved_image is your matrix
min_value = min(convolved_image(:));
max_value = max(convolved_image(:));

%%
%% Play with OR (try pi/2 or pi/4) and SF (try .05 or .1) 

%%% Part 2: Gabor Function

% a) Define values
OR = pi/4; % Orientation in radians which is pi/2
SF = 0.05; % Spatial frequency

% b) Create matrices
[x, y] = meshgrid(-20:20);

% Skew and rotate the Gaussian
X = x * cos(OR) + y * sin(OR);
Y = -x * sin(OR) + y * cos(OR);

% c) Create Gabor function
std_x = 7;
std_y = 17;

% 2D Gaussian
% gaussian = exp(-0.5 * ((X/std_x).^2 + (Y/std_y).^2));

gaussian = exp(-((X.^2 ./ (2.*std_x.^2)) + (Y.^2 ./ (2.*std_y.^2))));

% Modulate the Gaussian with a sine wave
A = sin(2.*pi.*SF.*X) ./ (2.*pi.*std_x.*std_y); % (2*pi*SF*X.*Y);

% Gabor function
gabor_function = A .* gaussian;

figure; imagesc(gabor_function); colormap gray; axis square;
title ('Gabor Function with OR = pi/4 and SF = .1')
xlabel('Horizontal Posisition')
ylabel('Vertical Posistion')
% temp plot out 
%  Plot the Gabor function
% surf(gabor_function);
% title('Gabor Function');

% imagesc(gabor_function)
% still step 2 
% e) Load 'rose.jpg' and convolve with Gabor function
% Load 'rose.jpg' and convert to grayscale
% rose_image = imread('C:\Users\jaspreetsingh\MATLAB-Drive\Systems Neuroscience\rose.jpg'); 
rose_image = imread('rose.jpg');
% rose=im2double(imread("rose.jpg"))
gabor_function = A .* gaussian;
gray_rose = double(rose_image);
convolved_image = conv2(gray_rose, gabor_function, 'same');

% Plot original image
figure;
sgtitle('Rose and Convolved Rose 3 with OR = pi/4 and SF =.05')
subplot(1, 2, 1);
imagesc(gray_rose);
colormap gray;
title('Original Image');
xlabel('Horizontal Position (pixel)');
ylabel('Vertical Position(pixel)');

% Plot result of convolution
% figure;
subplot(1, 2, 2);
imagesc(convolved_image);
colormap gray;
title ('Convolved Rose')
xlabel('Horizontal Position (pixel)');
ylabel('Vertical Position(pixel)');
% Assuming convolved_image is your matrix
min_value = min(convolved_image(:));
max_value = max(convolved_image(:));


%% rose 4 

%% Play with OR (try pi/2 or pi/4) and SF (try .05 or .1) 

%%% Part 2: Gabor Function

% a) Define values
OR = pi/4; % Orientation in radians which is pi/2
SF = 0.1; % Spatial frequency

% b) Create matrices
[x, y] = meshgrid(-20:20);

% Skew and rotate the Gaussian
X = x * cos(OR) + y * sin(OR);
Y = -x * sin(OR) + y * cos(OR);

% c) Create Gabor function
std_x = 7;
std_y = 17;

% 2D Gaussian
% gaussian = exp(-0.5 * ((X/std_x).^2 + (Y/std_y).^2));
gaussian = exp(-((X.^2 ./ (2.*std_x.^2)) + (Y.^2 ./ (2.*std_y.^2))));

% Modulate the Gaussian with a sine wave
A = sin(2.*pi.*SF.*X) ./ (2.*pi.*std_x.*std_y); % (2*pi*SF*X.*Y);

% Gabor function
gabor_function = A .* gaussian;

figure; imagesc(gabor_function); colormap gray; axis square;
title ('Gabor Function with OR = pi/4 and SF = .1')
xlabel('Horizontal Posisition')
ylabel('Vertical Posistion')
% temp plot out 
%  Plot the Gabor function
% surf(gabor_function);
% title('Gabor Function');

% imagesc(gabor_function)
% still step 2 
% e) Load 'rose.jpg' and convolve with Gabor function
% Load 'rose.jpg' and convert to grayscale
% rose_image = imread('C:\Users\jaspreetsingh\MATLAB-Drive\Systems Neuroscience\rose.jpg'); 
rose_image = imread('rose.jpg');
% rose=im2double(imread("rose.jpg"))
gabor_function = A .* gaussian;
gray_rose = double(rose_image);
convolved_image = conv2(gray_rose, gabor_function, 'same');

% Plot original image
figure;
sgtitle('Rose and Convolved Rose 4 with OR = pi/4 and SF =.1')
subplot(1, 2, 1);
imagesc(gray_rose);
colormap gray;
title('Original Image');
xlabel('Horizontal Position (pixel)');
ylabel('Vertical Position(pixel)');

% Plot result of convolution
% figure;
subplot(1, 2, 2);
imagesc(convolved_image);
colormap gray;
title ('Convolved Rose')
xlabel('Horizontal Position (pixel)');
ylabel('Vertical Position(pixel)');
% Assuming convolved_image is your matrix
min_value = min(convolved_image(:));
max_value = max(convolved_image(:));

%% rose convolved plot 3 with diff params
% imagesc(gabor_function)
% still step 2 
% e) Load 'rose.jpg' and convolve with Gabor function
% Load 'rose.jpg' and convert to grayscale
% rose_image = imread('C:\Users\jaspreetsingh\MATLAB-Drive\Systems Neuroscience\rose.jpg'); 
rose_image = imread('rose.jpg');
% rose=im2double(imread("rose.jpg"))
gabor_function = A .* gaussian;
gray_rose = double(rose_image);
convolved_image = conv2(gray_rose, gabor_function, 'same');

% Plot original image
figure;
sgtitle('Rose and Convolved Rose 3')
subplot(1, 2, 1);
imagesc(gray_rose);
colormap gray;
title('Original Image');
xlabel('Horizontal Position (pixel)');
ylabel('Vertical Position(pixel)');

% Plot result of convolution
% figure;
subplot(1, 2, 2);
imagesc(convolved_image);
colormap gray;
xlabel('Horizontal Position (pixel)');
ylabel('Vertical Position(pixel)');
% Assuming convolved_image is your matrix
min_value = min(convolved_image(:));
max_value = max(convolved_image(:));
% 
% % Display the image with the appropriate display range
% imshow(convolved_image, [min_value, max_value]);
% imshow(convolved_image, []);
% title('Convolved Image');

% % f) Play with OR and SF
% new_OR = pi/4; % Try different values
% new_SF = 0.05; % Try different values
% % % % % % % % % % % 
% % % % % % % % % % % % Update Gabor function with new orientation and spatial frequency
% % % % % % % % % % % new_X = x * cos(new_OR) + y * sin(new_OR);
% % % % % % % % % % % new_Y = -x * sin(new_OR) + y * cos(new_OR);
% % % % % % % % % % % new_A = sin(2*pi*new_SF*x) ./ (2*pi*new_SF*x.*y);
% % % % % % % % % % % new_gabor_function = new_A .* exp(-0.5 * ((new_X/std_x).^2 + (new_Y/std_y).^2));

% % % % % % % % % % % % % % Convolve with updated Gabor function
% % % % % % % % % % % % % new_convolved_image = conv2(gray_rose, new_gabor_function, 'same');
% % % % % % % % % % % % % 
% % % % % % % % % % % % % % Plot the result of convolution with new parameters
% % % % % % % % % % % % % figure;
% % % % % % % % % % % % % subplot(1, 2, 1);
% % % % % % % % % % % % % imagesc(convolved_image);
% % % % % % % % % % % % % colorbar; % Add a colorbar to show intensity scale

%% Step 3: Hgrid image and stimulated visual image

% 1) Load the image
hgrid = load('hgrid.mat');
image_hgrid = hgrid.hgrid;
%%

%% Step 3: retina function

% Load image
% load('hgrid.mat');
imshow(hgrid);
title("Hgrid Origional Image")
xlabel ("horozontal posistion")
ylabel ('vertical posistion')

% Define parameters
size_fraction = 0.1;
kernel_std = 0.15;

% Call retina function
retina_output_with_dots = retina(size_fraction, kernel_std, hgrid);
figure;
subplot(1, 2, 1);
colormap gray;
imagesc(retina_output_with_dots);
title('Illusory Gray Dots');

% Adjust parameters to make dots disappear
size_fraction = 0.04;
retina_output_without_dots = retina(size_fraction, kernel_std, hgrid);
subplot(1, 2, 2);
imagesc(retina_output_without_dots);
colormap gray;
title('No Illusory Gray Dots');

% Updated retina function
function im = retina(pct, sd, image)
    imSize = max(size(image));
    rfSize = round(pct * imSize);
    dog = fspecial('log', rfSize, rfSize * sd);
    c = -filter2(dog, image); % switch the sign here for an off-center cell
    % these two lines rescale c between 0 and 1
    c = c - min(min(c)); 
    c = c / max(max(c));
  
    im = c;
end

%% adjusting parameters 


%title ('Adjusted Parameter No Gray Dot 1') 

%

