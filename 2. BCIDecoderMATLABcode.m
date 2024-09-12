% Jaspreet Singh
% Assignment III: BCI 
%Quantitative Systems Neurosience 
%1/31/24

%% Part 1: Continuous decoding

% Load data
load('continuous1.mat');

%% step 1 - downsampling new 
% Assuming N is the downsampling factor
N = 222;  % Change this to your desired downsampling factor

% Downsample the rate data
downsampled_rate = downsample(rate, N);

% Downsample the kinematic data accordingly
downsampled_kin = downsample(kin, N);
% step 1
% Plot the hand movements using an easier function like scatter to
% visualize data
figure;
comet(downsampled_rate(:, 1), downsampled_kin(:, 2)); % Scatter plot with blue filled circles
title('Monkey Hand Movements during Random Target Tracking');
xlabel('X Coordinate');
ylabel('Y Coordinate');
axis equal; % Ensure equal scaling on both axes for accurate representation
%%
%% step 1 - downsampling new 
% Assuming N is the downsampling factor
N = 100;  % Change this to your desired downsampling factor

% Downsample the rate data
downsampled_rate = downsample(rate, N);

% Downsample the kinematic data accordingly
downsampled_kin = downsample(kin, N);
% step 1
% Plot the hand movements using an easier function like scatter to
% visualize data
figure;
comet(downsampled_rate(:, 1), downsampled_kin(:, 2)); % Scatter plot with blue filled circles
title('Monkey Hand Movements during Random Target Tracking');
xlabel('X Coordinate');
ylabel('Y Coordinate');
axis equal; % Ensure equal scaling on both axes for accurate representation

%% step 2
% Compute linear filter coefficients
X = [rate, ones(size(rate, 1), 1)];  % Append a column of ones
Y = circshift(kin, -2, 1);  % Time-shifted kinematic data
A = (X' * X) \ (X' * Y);  % Linear regression to find coefficients

% Test the decoder's performance
decoded_kin = X * A;
mse = mean((decoded_kin - Y).^2, 'all');  % Mean Squared Error
% use correlation here

% Display results
figure;
subplot(2, 1, 1);
plot(Y(:, 1), 'b', 'LineWidth', 1.5);
hold on;
plot(decoded_kin(:, 1), 'r--', 'LineWidth', 1.5);
title('X-coordinate Reconstruction');
xlabel('Time');
ylabel('Position');
legend('Actual', 'Reconstructed');
% plot(Y(:, 1) - decoded_kin(:, 1)); % plot residuals

subplot(2, 1, 2);
plot(Y(:, 2), 'b', 'LineWidth', 1.5);
hold on;
plot(decoded_kin(:, 2), 'r--', 'LineWidth', 1.5);
title('Y-coordinate Reconstruction');
xlabel('Time');
ylabel('Position');
legend('Actual', 'Reconstructed');
% plot(Y(:, 2) - decoded_kin(:, 2)); % plot residuals

fprintf('Mean Squared Error: %.4f\n', mse);

%%% Compute linear filter coefficients
X = [rate, ones(size(rate, 1), 1)];  % Append a column of ones
Y = circshift(kin, -2, 1);  % Time-shifted kinematic data
A = (X' * X) \ (X' * Y);  % Linear regression to find coefficients

% Test the decoder's performance
decoded_kin = X * A;
mse = mean((decoded_kin - Y).^2, 'all');  % Mean Squared Error

% Display results
figure;
subplot(2, 1, 1);
plot(Y(:, 1), 'b', 'LineWidth', 1.5);
hold on;
plot(decoded_kin(:, 1), 'r--', 'LineWidth', 1.5);
title(['X-coordinate Reconstruction (MSE = ', num2str(mse), ')']);  % Display MSE in the title
xlabel('Time');
ylabel('Position');
legend('Actual', 'Reconstructed');

subplot(2, 1, 2);
plot(Y(:, 2), 'b', 'LineWidth', 1.5);
hold on;
plot(decoded_kin(:, 2), 'r--', 'LineWidth', 1.5);
title(['Y-coordinate Reconstruction (MSE = ', num2str(mse), ')']);
xlabel('Time');
ylabel('Position');
legend('Actual', 'Reconstructed');

% Add MSE to the x-axis title
subplot(2, 1, 1);
ax1 = gca; % Get current axis
title(ax1, {ax1.Title.String, ['MSE: ', num2str(mse)]});

fprintf('Mean Squared Error: %.4f\n', mse);


%% Part 1 - Real-time decoding

% Load new data
%load('continuous2.mat');

% Test on new data
X_new = [rate, ones(size(rate, 1), 1)];
Y_new = circshift(kin, -2, 1);
decoded_kin_new = X_new * A;

% Evaluate performance
mse_new = mean((decoded_kin_new - Y_new).^2, 'all');

% Display results for new data
figure;
subplot(2, 1, 1);
plot(Y_new(:, 1), 'b', 'LineWidth', 1.5);
hold on;
plot(decoded_kin_new(:, 1), 'r--', 'LineWidth', 1.5);
title('X-coordinate Reconstruction (New Data)');
xlabel('Time');
ylabel('Position');
legend('Actual', 'Reconstructed');

subplot(2, 1, 2);
plot(Y_new(:, 2), 'b', 'LineWidth', 1.5);
hold on;
plot(decoded_kin_new(:, 2), 'r--', 'LineWidth', 1.5);
title('Y-coordinate Reconstruction (New Data)');
xlabel('Time');
ylabel('Position');
legend('Actual', 'Reconstructed');

fprintf('Mean Squared Error (New Data): %.4f\n', mse_new);
%%
% Load new data
%load('continuous2.mat');

% Test on new data
X_new = [rate, ones(size(rate, 1), 1)];
Y_new = circshift(kin, -2, 1);
decoded_kin_new = X_new * A;

% Evaluate performance
mse_new_x = mean((decoded_kin_new(:,1) - Y_new(:,1)).^2, 'all');
mse_new_y = mean((decoded_kin_new(:,2) - Y_new(:,2)).^2, 'all');

% Display results for new data
figure;
subplot(2, 1, 1);
plot(Y_new(:, 1), 'b', 'LineWidth', 1.5);
hold on;
plot(decoded_kin_new(:, 1), 'r--', 'LineWidth', 1.5);
title(['X-coordinate Reconstruction (New Data) (MSE = ', num2str(mse_new_x), ')']); % Display MSE in the title
xlabel('Time');
ylabel('Position');
legend('Actual', 'Reconstructed');

subplot(2, 1, 2);
plot(Y_new(:, 2), 'b', 'LineWidth', 1.5);
hold on;
plot(decoded_kin_new(:, 2), 'r--', 'LineWidth', 1.5);
title(['Y-coordinate Reconstruction (New Data) (MSE = ', num2str(mse_new_y), ')']); % Display MSE in the title
xlabel('Time');
ylabel('Position');
legend('Actual', 'Reconstructed');

fprintf('Mean Squared Error (New Data - X): %.4f\n', mse_new_x);
fprintf('Mean Squared Error (New Data - Y): %.4f\n', mse_new_y);


%% Part 1 - Explore different time lags

time_lags = [210, 140, 70, 0, -70, -140, -210]; % shift behavior BACK in time with negative numbers
ms_per_sample = 70;
mses = nan(numel(time_lags), 1);
figure;
for i = 1:length(time_lags)
    Y_shifted = circshift(kin, round(time_lags(i)/ms_per_sample), 1);
    X_shifted = X_new;
    A_shifted = (X_shifted' * X_shifted) \ (X_shifted' * Y_shifted);
    decoded_kin_shifted = X_shifted * A_shifted;
    mse_shifted = mean((decoded_kin_shifted - Y_shifted).^2, 'all');
    mses(i) = mse_shifted;

    subplot(numel(time_lags), 2, 2 * i - 1);
    plot(Y_shifted(:, 1), 'b', 'LineWidth', 1.5);
    hold on;
    plot(decoded_kin_shifted(:, 1), 'r--', 'LineWidth', 1.5);
    title(['X-coordinate Reconstruction (Time Lag = ' num2str(time_lags(i)) ' ms), MSE = ', num2str(mse_shifted)]);
    xlabel('Time');
    ylabel('Position');
    legend('Actual', 'Reconstructed');

    subplot(numel(time_lags), 2, 2 * i);
    plot(Y_shifted(:, 2), 'b', 'LineWidth', 1.5);
    hold on;
    plot(decoded_kin_shifted(:, 2), 'r--', 'LineWidth', 1.5);
    title(['Y-coordinate Reconstruction (Time Lag = ' num2str(time_lags(i)) ' ms)']);
    xlabel('Time');
    ylabel('Position');
    legend('Actual', 'Reconstructed');
end

% figure; plot(time_lags, mses);

% %% Part 1 - Bonus Question (Multiple Time Lags)
% 
% % Implementing a model with multiple time lags
% 
% num_time_lags = 5;  % Adjust as needed
% 
% % Create X matrix with multiple time lags
% X_multi_lag = zeros(size(X, 1) - num_time_lags, size(X, 2) * num_time_lags);
% for lag = 0:num_time_lags - 1
%     X_multi_lag(:, (1:size(X, 2)) + lag * size(X, 2)) = X(num_time_lags - lag:end - 1 - lag, :);
% end
% 
% % Solve for coefficients
% A_multi_lag = (X_multi_lag' * X_multi_lag) \ (X_multi_lag' * Y(num_time_lags:end, :));
% 
% % Test the decoder's performance with multiple time lags
% decoded_kin_multi_lag = X_multi_lag * A_multi_lag;
% mse_multi_lag = mean((decoded_kin_multi_lag - Y(num_time_lags:end, :)).^2, 'all');
% 
% % Display results
% figure;
% subplot(2, 1, 1);
% plot(Y(num_time_lags:end, 1), 'b', 'LineWidth', 1.5);
% hold on;
% plot(decoded_kin_multi_lag(:, 1), 'r--', 'LineWidth', 1.5);
% title(['X-coordinate Reconstruction (Multiple Time Lags)']);
% xlabel('Time');
% ylabel('Position');
% legend('Actual', 'Reconstructed');
% 
% subplot(2, 1, 2);
% % plot(Y(num_time_lags:end, 2), 'b', 'LineWidth', 1.5


% %% Part 2: Discrete decoding
% 
% 
% % Load spikeCounts data
% % load('spikeCounts.mat');
% 
% % Initialize Results structure
% numReps = size(SpikeCounts, 1);
% numTargs = size(SpikeCounts, 3);
% 
% Results(numTargs, numReps).actualTarg = [];
% Results(numTargs, numReps).decodedTarg = [];
% 
% % Visualize the tuning of cells (average spike count for 5 targets)
% figure;
% for neuronInd = 1:size(SpikeCounts, 2)
%     subplot(4, 5, min(neuronInd, 20));
%     avgSpikeCount = squeeze(mean(SpikeCounts(:, neuronInd, :), 1));
%     bar(1:numTargs, avgSpikeCount);
%     title(['Neuron ' num2str(neuronInd)]);
%     xlabel('Target');
%     ylabel('Average Spike Count');
% end
% sgtitle('Average Spike Count for Neurons Across Targets');
% 
% % Leave-one-out cross-validation Github
%   % % Select training and testing data (you can choose to split your data in a different way if you wish)
%   %   train = trial(ix(1:split),:);
%   %   test= trial(ix(split+1:end),:);
%   % 
%   %   [F,l,t,mF]=organize_data1(train,dt,320);
%   %   [F_test,l_test,t_test]=organize_data1(test,dt,320);
% 
% 
% % Leave-one-out cross-validation
% for testRep = 1:numReps
%     trainReps = setdiff(1:numReps, testRep);
% 
%     % Train the decoder
%     Lambda = zeros(numTargs, size(SpikeCounts, 2));
%     for targInd = 1:numTargs
%         for neuronInd = 1:size(SpikeCounts, 2)
%             Lambda(targInd, neuronInd) = mean(SpikeCounts(trainReps, neuronInd, targInd));
%         end
%     end
% 
%     % Test the decoder on the trial set aside
%     obsSpikeCount = squeeze(SpikeCounts(testRep, :, :));
%     [~, decodedTarg] = max(obsSpikeCount .* log(Lambda'), [], 2);
% 
%     % Store the actual and decoded targets
%     for targIdx = 1:numTargs
%         % Use targInd instead of for-loop variable
%         Results(targIdx, testRep).actualTarg = targInd;
%     end
%     Results(:, testRep).decodedTarg = decodedTarg;
% end
% 
% %%
% % Plot the confusion matrix
% ConfuMat = plotConfusion(Results);
% title('Confusion Matrix');
% 

% 
%% Part 2: attempt 3 using the leave out method to help make predictions for correlations to go into the Rp and Rc equation

% Load spikeCounts data
load('spikeCounts.mat');

% Initialize Results structure
numReps = size(SpikeCounts, 1);
numNeurons = size(SpikeCounts, 2);
numTargets = size(SpikeCounts, 3);

Results(numTargets, numReps).actualTarg = [];
Results(numTargets, numReps).decodedTarg = [];

% Visualize the tuning of cells (average spike count for 5 targets)
figure;
for neuronInd = 1:numNeurons
    subplot(4, 5, min(neuronInd, 20));
    avgSpikeCount = squeeze(mean(SpikeCounts(:, neuronInd, :), 1));
    bar(1:numTargets, avgSpikeCount);
    title(['Neuron ' num2str(neuronInd)]);
    xlabel('Target');
    ylabel('Average Spike Count');
end
sgtitle('Average Spike Count for Neurons Across Targets');

% SpikeCounts = zscore(SpikeCounts, 0, [1, 2]);

% Results = nan(numTargets * numReps, 2); rowCount = 1;
% Leave-one-out cross-validation
for testTarg = 1:numTargets
    for testRep = 1:numReps
    
   

        obsSpikeCount = squeeze(SpikeCounts(testRep, :, testTarg));
        trainData = SpikeCounts;
        trainData(testRep, :, testTarg) = NaN;
        Lambda = squeeze(mean(trainData, 1, "omitnan"))';
    
    
        % Find the target that is most likely given the observed neural activity
        targetLikelihoods = log(Lambda + 1) * obsSpikeCount'; % need to z-score along targets
        [~, decodedTarg] = max(targetLikelihoods);

        targetDists = pdist2(Lambda, obsSpikeCount);
        [~, decodedTarg] = min(targetDists);

        % correlation

        % Results(rowCount, :) = [testTarg, decodedTarg]; % rowCount = rowCount + 1;
        Results(testTarg, testRep).actualTarg = testTarg;
        Results(testTarg, testRep).decodedTarg = decodedTarg;
    
    end
end

ConfuMat = plotConfusion(Results); colorbar;
title('Confusion Matrix');

%% attempt 3 
% Plot the confusion matrix
ConfuMat = plotConfusion(Results);
title('Confusion Matrix');

% Initialize matrices to store correlations
rp = zeros(numReps, numNeurons, numTargets);
rc = zeros(numReps, numNeurons, numTargets);

% Loop over each repetition and neuron
for testRep = 1:numReps
    for neuronInd = 1:numNeurons
        % Loop over each target
        for targInd = 1:numTargets
            % Calculate correlation between observed spike counts and component predictions
            rp(testRep, neuronInd, targInd) = corr(squeeze(SpikeCounts(testRep, neuronInd, targInd)), ...
                squeeze(componentPredictions(testRep, targInd, :))); % Corrected indexing

            % Calculate correlation between observed spike counts and pattern predictions
            rc(testRep, neuronInd, targInd) = corr(squeeze(SpikeCounts(testRep, neuronInd, targInd)), ...
                squeeze(patternPredictions(testRep, :))); % Corrected indexing
        end
    end
end


% Display results
disp('Mean correlation between observed spike counts and component predictions:')
mean_rp = mean(rp(:))
disp('Mean correlation between observed spike counts and pattern predictions:')
mean_rc = mean(rc(:))
%%
% Compute accuracy, precision, recall, and F1-score from confusion matrix
total_samples = sum(ConfuMat(:));
true_positives = diag(ConfuMat);
false_positives = sum(ConfuMat, 1) - true_positives';
false_negatives = sum(ConfuMat, 2) - true_positives;

% Compute accuracy
accuracy = sum(true_positives) / total_samples;

% Compute precision
precision = true_positives ./ (true_positives + false_positives);

% Compute recall (sensitivity)
recall = true_positives ./ (true_positives + false_negatives);

% Compute F1-score
f1_score = 2 * (precision .* recall) ./ (precision + recall);

% Display performance metrics
disp('Performance Metrics:');
disp(['Accuracy: ', num2str(accuracy)]);
disp(['Precision: ', num2str(mean(precision))]);
disp(['Recall (Sensitivity): ', num2str(mean(recall))]);
disp(['F1-Score: ', num2str(mean(f1_score))]);


