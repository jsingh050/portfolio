%Jaspreet Singh
%Spike Sorting neural data 
%January 9, 2023

%% load the dataset
% data variable "data" has two structures: wf is waveforms and stamps is
% timestamps

load waveforms.mat

%% Step 1: plotting the waveforms  
% plot 100 waveforms 
    figure;
    %set the background to white 
    set(gca, 'Color', 'w');
    idx = floor(linspace(1, size(data.wf, 1), 100));
    for i = 1:100
        subplot (10, 10, i);
        plot(data.wf(idx(i),:));
        title (['Waveform ' num2str(idx(i))]);
        % Labeling x and y axes for each subplot
        xlabel('Time (ms)');
        ylabel('Amplitude (mV)');
    end
%%plot overlapped waveforms 
 
figure;
set(gca, 'Color', 'w');
% Plot all 100 waveforms on the same axes
hold on;
for i = 1:100
    plot(data.wf(i, :), 'Color', [rand(), rand(), rand()]);  % Use different colors for each waveform
end
hold off;
title('Overlapping Waveforms');
xlabel('Time (ms)');
ylabel('Amplitude (mV)');
%% Step 2: doing PCA 

% Perform PCA
coeff = pca(data.wf);
pc = data.wf * coeff(:, 1:3);

% Plot the projections onto the first two PCs
figure;inc
scatter(pc(:, 1), pc(:, 2), 10, '.');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
zlabel('Principal Component 3');
title('Projections onto First Two PCs');

figure;
X = -400:40:400;
Y = -400:40:400;
h = histogram2(pc(:, 1), pc(:, 2), X, Y);
surf(X(1:end-1), Y(1:end-1), h.Values); colormap hot;
% scatter(pc(:, 1), pc(:, 2), 10, pc(:, 3)); colormap cool;

%%

% Extract waveforms
waveforms = data.wf;

% % Reshape the waveforms to have each row as a data point
% reshaped_waveforms = reshape(waveforms, size(waveforms, 1), []);

% Perform PCA
[coeff, score, latent, ~, explained] = pca(reshaped_waveforms);

% Plot the explained variance
figure;
plot(cumsum(explained), 'bo-');
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('Cumulative Explained Variance');

% Choose the number of principal components (e.g., 2 for 2D plot)
num_components = 2;

% Plot the projections onto the first two principal components
projection = score(:, 1:num_components);
figure;
scatter(projection(:, 1), projection(:, 2));
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('Projections onto the First Two Principal Components');

%for 3D plot 
num_components = 3;
projection = score(:, 1:num_components);
figure;
scatter3(projection(:, 1), projection(:, 2), projection(:, 3));
xlabel('Principal Component 1');
ylabel('Principal Component 2');
zlabel('Principal Component 3');
title('Projections onto the First Three Principal Components');


% 
% 
% % visually estimate the number of neurons based on the plot
% %
% 
% %% Step 4: Plot projections onto the first two PCs
% figure;
% scatter(pc(:, 1), pc(:, 2), 10, '.');
% xlabel('Principal Component 1');
% ylabel('Principal Component 2');
% title('Projections onto First Two PCs');
% 
% %% Step 5: Apply k-means clustering
% k = 5; % Set k to the estimated number of neurons
% [idx, centroids] = kmeans(pc, k);
% 
% % Load the dataset
% load('waveforms.mat');

%% PCA Part 1

% Perform PCA
coeff = pca(data.wf);
pc = data.wf * coeff(:, 1:15); % use any number of PCs for clustering

% Set the number of clusters (k)
k = 2;  % Adjust this based on your estimate

% Apply k-means clustering
[idx, centroids] = kmeans(pc, k);

% Plot the results
figure;
cmap = hsv(k);
scatter(pc(:, 1), pc(:, 2), 10, cmap(idx, :), '.');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('K-Means Clustering');

% Display cluster centroids
hold on;
scatter(centroids(:, 1), centroids(:, 2), 100, 'k', 'x');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Centroids');
hold off;
%%
% Step 5: Rotate unit vectors back to original space using SVD

% Perform PCA
[coeff, score] = pca(data.wf);

% Number of principal components to consider
num_components = 2;

% Initialize figure
figure; hold on;

for i = 1:num_components
    % Take a unit vector along the ith principal component
    uv = zeros(1, size(data.wf, 2));
    uv(i) = 1;

    % Rotate the unit vector back to the original space
    back = uv * coeff';

    % Plot the waveform
    subplot(num_components, 1, i);
    plot(1:size(data.wf, 2), back);
    title(['PC ' num2str(i)]);
    xlabel('Time (ms)');
    ylabel('Amplitude (A)');
end

hold off;

%% Step 6: Plot 100 waveforms 

figure;
cidx = floor(linspace(1, size(data.wf, 1), 100));
cmap = hsv(k);
for i = 1:100
    subplot(10, 10, i);
    plot(data.wf(cidx(i), :), 'color', cmap(idx(cidx(i)), :));
    title(['Waveform ' num2str(cidx(i))]);
        % Labeling x and y axes for each subplot
        xlabel('Time (ms)');
        ylabel('Amplitude (A)');
end
%% 
% Step 7: Plot all waveforms color-coded by k-means clustering

% Perform PCA
coeff = pca(data.wf);
pc = data.wf * coeff(:, 1:15); % use any number of PCs for clustering

% Set the number of clusters (k)
k = 2;  % determined earlier 

% Apply k-means clustering
[idx, centroids] = kmeans(pc, k);

% Color-coded plotting for all waveforms
cmap = hsv(k);
figure;

for i = 1:size(data.wf, 1)
    plot(data.wf(i, :), 'color', cmap(idx(i), :));
    hold on;
end

title('Color-Coded Waveforms by PCA Clusters');
xlabel('Time (ms)');
ylabel('Amplitude (A)');

hold off;



%% Step 5: Rotate unit vector back to original space

[coeff, score] = pca(data.wf);
% spkid = 3;
uv = [1; zeros(size(data.wf, 2) - 1, 1)]';
% uv = score(spkid, :);
back = uv*coeff';

figure; hold on;
plot(1:48, back)
plot(1:48, uv);

%svd and get u compoennt of u and then second component of v 
%(u, v, :) == the all of the first component and all of the second component 

%% plot PCA elbow (explained variance)

% Perform PCA
[coeff, score, latent, ~, explained] = pca(data.wf);

% Plot the explained variance
figure;
plot(cumsum(explained), 'bo-');
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('Cumulative Explained Variance');

%% Step 8: plot color-coded rasters

% Perform PCA
coeff = pca(data.wf);
pc = data.wf * coeff(:, 1:15); % use any number of PCs for clustering

% Set the number of clusters (neurons)
num_clusters = 2;  % Adjust as needed

% Perform k-means clustering
idx = kmeans(pc, num_clusters);

timetoplot = 1000:1100; % seconds
spikesidx = find(data.stamps >= timetoplot(1) & data.stamps <= timetoplot(end));
colors = hsv(num_clusters);
figure; hold on;
for s = 1:numel(spikesidx)
    t = data.stamps(spikesidx(s));
    c = idx(spikesidx(s));
    plot([t, t], [0, 1], "Color", colors(c, :), "LineWidth", 1);
end
xlim([timetoplot(1), timetoplot(end)]);

%%
% Perform PCA
coeff = pca(data.wf);
pc = data.wf * coeff(:, 1:15); % use any number of PCs for clustering
% Set the number of clusters (neurons)
num_clusters = 2;  % Adjust as needed
% Perform k-means clustering
idx = kmeans(pc, num_clusters);
timetoplot = 1000:1100; % seconds
spikesidx = find(data.stamps >= timetoplot(1) & data.stamps <= timetoplot(end));
colors = hsv(num_clusters);

% Create new figure for the first raster
figure;
subplot(2,1,1); % Create subplot for the first raster
hold on;
for s = 1:numel(spikesidx)
    t = data.stamps(spikesidx(s));
    c = idx(spikesidx(s));
    plot([t, t], [0, 1], "Color", colors(c, :), "LineWidth", 1);
end
xlim([timetoplot(1), timetoplot(end)]);
title('Raster 1'); % Add title for the first raster

% Create new subplot for the second raster
subplot(2,1,2); % Create subplot for the second raster
hold on;

% Add title and labels for the second raster
title('Raster 2'); % Add title for the second raster
xlabel('Time (seconds)'); % Add x-axis label
ylabel('Neuron'); % Add y-axis label

%% ISI histogram

% Perform PCA
coeff = pca(data.wf);
pc = data.wf * coeff(:, 1:15); % use any number of PCs for clustering

% Set the number of clusters (neurons)
num_clusters = 2;  % Adjust as needed

% Perform k-means clustering
idx = kmeans(pc, num_clusters);

figure; tiledlayout(1, num_clusters);
for cl = 1:num_clusters
    ISI = diff(data.stamps(idx == cl));
    nexttile; histogram(ISI, binedges=0:0.002:1); % 
    xlabel('Interspike Intervals'); % X-axis label
ylabel('Count'); % Y-axis label
title('Interspike Intervals Histogram'); 
end
