 %Jaspreet Singh
%Assignment 4 (MT Vision Assignment) 
%Feb 9,2024
%Quantitative Systems Neuroscience 

%%Step 1 - Correct part 2 with the actual sigmoid curve 
% Defining the Weibull function for the psychometric curve fitting
p = @(b, x) 1 - 0.5 * exp(-(x / b(1)).^b(2));

% Define coherence levels
coherences = [0, 4, 8, 16, 32, 64];

% Initialize an array to hold mean correct responses for each coherence
correct_per_coh = zeros(1, length(coherences));

% Retrieve the correct response rates for each coherence level
for i = 1:length(coherences)
    correct_responses = MTdata(:,:,i);
    correct_per_coh(i) = mean(correct_responses(:,3));
end

% Parameters for the Weibull function fit - initial guesses
% These values are typical starting points, but may need adjustment
alpha_guess = 5;  % Guess for the alpha parameter
beta_guess = 1;    % Guess for the beta parameter
guess = [alpha_guess, beta_guess];  % Vector of initial guesses

% Fit the Weibull function to the data
[ab_fit, ~] = nlinfit(coherences, correct_per_coh, p, guess);

% Create a fine coherence level array for plotting the fit line
fine_coherence = linspace(0, max(coherences), 100);
fit_line = 1 - 0.5 * exp(-(fine_coherence / ab_fit(1)).^ab_fit(2));

% Plot the data and the fitted Weibull psychometric curve
figure;
scatter(coherences, correct_per_coh, 75, 'filled', 'k'); % Plot the actual data points
hold on;
plot(fine_coherence, fit_line, 'r', 'LineWidth', 2); % Plot the psychometric curve
% Now, mark each of the six coherence levels with a specific marker
for c = 1:length(coherences)
    plot(coherences(c), correct_per_coh(c), 'bo', 'MarkerSize', 10, 'LineWidth', 2);
    text(coherences(c), correct_per_coh(c), sprintf('  %d%%', coherences(c)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
end
xlabel('Coherence Level (%)');
ylabel('Proportion Correct');
title('Psychometric Curve of Choice Data with Weibull Fit');
legend('Data Points','Weibull Fit', 'Coherence Levels', 'Location', 'best');

hold off;




%% Step 2 Construct a neurometric curve from the pair of neurons
%alpha and beta are difficult to guess, previous research (1996) offers
%insight 
alpha = 1;
beta = 0.1;

% Define the Weibull function
fun = @(params, c) 1 - 0.5 * exp(-(c / params(1)).^params(2)); % Weibull function

% Initialize variables to store AUC measurements
coherence_levels = coherence; % Example coherence levels
auc_measurements = zeros(size(coherence_levels));

% Iterate over each coherence level
for i = 1:length(coherence_levels)
    
    % Calculate null and preferred distributions of firing rates
    null_distribution = MTdata(:, 1, i); % Assuming first dimension represents coherence, 
    % second dimension represents neuron, and third dimension represents null (1) or preferred (2)
    preferred_distribution = MTdata(:, 2, i);
    
    % Slide criterion from 0 to 100
    criteria = 0:100;
    auc = zeros(numel(criteria), 2);
    
    % Calculate the percentage of firing rates greater than each criterion for both null and preferred distributions
    for j = 1:length(criteria)
        criterion = criteria(j);
        null_percentage = sum(null_distribution > criterion) / numel(null_distribution);
        preferred_percentage = sum(preferred_distribution > criterion) / numel(preferred_distribution);
        auc(j, :) = [null_percentage, preferred_percentage];
    end

    % Calculate the area under the curve (AUC) using trapz (aka perform
    % numerical integration) 
    auc_measurements(i) = -trapz(auc(:, 1), auc(:, 2));
    % if auc_measurements(i) < 0
    %     auc_measurements(i) = 0;
    % end
end

% Fit the AUC measurements with the Weibull function
beta0 = [-3, 3]; % Initial guess for alpha and beta
[parameters, ~, ~, ~, ~] = nlinfit(coherence_levels, auc_measurements, fun, [alpha, beta]);

% Extract estimates of alpha and beta
alpha_auc = parameters(1);
beta_auc = parameters(2);
disp(['Alpha (Threshold): ', num2str(alpha_auc)]);
disp(['Beta (Sensitivity to Coherence): ', num2str(beta_auc)]);


figure;
scatter(coherence_levels, auc_measurements);
hold on;
plot(coherence_levels, fun(parameters, coherence_levels), 'r');
xlabel('Coherence');
ylabel('AUC');
title('Psychometric Curve - Choice Data and Fitted Weibull');
legend('Data', 'Fitted Weibull');
hold off;

%% Step 2 - Construct a Neurometric Curve from the Pair of Neurons

% Insights from previous research for initial guesses
alpha_guess = 0.061;
beta_guess = 1.17;

% Define the Weibull function
weibullFun = @(params, c) 1 - 0.5 * exp(-(c / params(1)).^params(2));

% Coherence levels for which data is available
coherence_levels = coherence; % Assuming 'coherence' is previously defined with coherence levels

% Prepare variables to store AUC for each coherence level
auc_measurements = zeros(size(coherence_levels));

% Loop over each coherence level to calculate AUC
for i = 1:length(coherence_levels)
    
    % Extract null and preferred distributions of firing rates for current coherence level
    null_distribution = MTdata(:, 1, i); % Assuming 'MTdata' is structured accordingly
    preferred_distribution = MTdata(:, 2, i);
    
    % Initialize an array to store the difference in percentage of firing rates
    auc = zeros(1, 101); % 101 because criteria range from 0 to 100 inclusive
    
    % Slide the criterion from 0 to 100
    for criterion = 0:100
        null_percentage = mean(null_distribution > criterion);
        preferred_percentage = mean(preferred_distribution > criterion);
        auc(criterion + 1) = preferred_percentage - null_percentage; % +1 for zero-based indexing
    end
    
    % Compute AUC using numerical integration
    auc_measurements(i) = trapz(auc);
    % Ensure AUC is non-negative
    auc_measurements(i) = max(auc_measurements(i), 0);
end

% Fit AUC measurements with the Weibull function using nonlinear curve fitting
initialGuess = [alpha_guess, beta_guess]; % Initial guesses for fitting
[parameters_est, ~] = nlinfit(coherence_levels, auc_measurements, weibullFun, initialGuess);

% Extract fitted alpha and beta
alpha_auc = parameters_est(1);
beta_auc = parameters_est(2);
disp(['Alpha (Threshold): ', num2str(alpha_auc)]);
disp(['Beta (Sensitivity to Coherence): ', num2str(beta_auc)]);

% Plotting the results
figure;
scatter(coherence_levels, auc_measurements, 'filled', 'DisplayName', 'AUC Data');
hold on;
fittedCurve = weibullFun(parameters_est, coherence_levels);
plot(coherence_levels, fittedCurve, 'r-', 'DisplayName', 'Fitted Weibull Curve');
xlabel('Coherence Level');
ylabel('Area Under Curve (AUC)');
title('Neurometric Curve based on AUC and Fitted Weibull Function');
legend('show');
hold off;

%% roc first plot 
hold on
for i=1:6
    plot(p_null(:,i),p_pref(:,i))
end
title('ROC Analysis')
xlabel('% Null > Criterion')
ylabel('% Preferred > Criterion')

%% creating the neurometric curve (IAN) 

%define Weibull function
p=@(b,x) 1-.5*exp(-(x/b(1)).^b(2));
%initialize values for alpha and beta
init=[10,2];

%fit using nonlinear regression
[b_fit1,~]=nlinfit(coherence*100,auc,p,init);

line_fit=1-.5*exp(-(linspace(1,70,100)/b_fit1(1)).^b_fit1(2));

fig3=figure(3);
semilogx(linspace(1,70,100),line_fit,'Color','r','LineWidth',1)
hold on
plot(coherence*100+1,auc,'o','MarkerEdgeColor','b')
xlabel('Coherence Level')
ylabel('Area Under Curve (AUC)')
title('Neurometric Curve')

legend({'Weibull Function','Data'},"Location","northwest")

exportgraphics(fig3,'fig3.png','Resolution',300)

%% Step 2 creating two ROC 
% Assuming you have null_responses and preferred_responses for each coherence level
% Assuming you have null_responses and preferred_responses for each coherence level

coherence_levels = coherence; % Assuming coherence levels are defined
num_coherences = numel(coherence_levels);

% Create a figure for the ROC curves
figure;

% Initialize arrays to store FPR and TPR for all neurons at each coherence level
all_fpr_left = cell(num_coherences, 1);
all_tpr_left = cell(num_coherences, 1);
all_fpr_right = cell(num_coherences, 1);
all_tpr_right = cell(num_coherences, 1);

% Loop through each coherence level
for i = 1:num_coherences
    % Extract null and preferred responses for current coherence level
    null_responses = MTdata(:, 1, i); % Assuming MTdata contains responses for left neuron
    preferred_responses = MTdata(:, 2, i); % Assuming MTdata contains responses for right neuron
    
    % Create true class labels for null and preferred responses
    % Label 0 for null responses, 1 for preferred responses
    true_labels_null = zeros(size(null_responses));
    true_labels_pref = ones(size(preferred_responses));
    
    % Combine true class labels and responses for each neuron
    labels = [true_labels_null; true_labels_pref];
    responses = [null_responses; preferred_responses];
    
    % Compute ROC curve for each neuron
    [fpr_left, tpr_left, ~] = perfcurve(labels, responses, 0);
    [fpr_right, tpr_right, ~] = perfcurve(labels, responses, 1);
    
    % Store FPR and TPR for left and right neurons
    all_fpr_left{i} = fpr_left;
    all_tpr_left{i} = tpr_left;
    all_fpr_right{i} = fpr_right;
    all_tpr_right{i} = tpr_right;
end

% Plot all ROC curves in one plot
hold on;
for i = 1:num_coherences
    plot(all_fpr_left{i}, all_tpr_left{i}, 'b', 'LineWidth', 1.5);
    plot(all_fpr_right{i}, all_tpr_right{i}, 'r', 'LineWidth', 1.5);
end
hold off;

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves for Left and Right Neurons at Different Coherence Levels');
legend('Left Neuron', 'Right Neuron', 'Location', 'southeast');
grid on;


%% Step 3: Generate a histogram of the 12 values you get (6 coherences * 2 neurons) and call the mean of that distribution the "choice probability"for this monkey.

% Define variables
num_coherences = 6; %given in assignment 
num_neurons = 2; %left motion and right motion neurons 
AUC_values = zeros(num_coherences, num_neurons);

total_left = zeros(num_coherences, 1); total_right = zeros(num_coherences, 1);

% forloop for the coherence
% for each neuron
for i = 1:num_coherences
    total_left(i) = nnz(MTdata(:, 3, i) == 0);
    total_right(i) = nnz(MTdata(:, 3, i) == 1);
    for j = 1:num_neurons
        % Split data into rightward and leftward choices
        rightward_choices = MTdata(MTdata(:, 3, i) == 1, j, i); % Assuming data structure is [coherence, neuron, choice, trial]
        leftward_choices = MTdata(MTdata(:, 3, i) == 0, j, i); %repeated for Neuron 2 or leftward choice
        
        % Slide criterion from 0 to 100
        criteria = 0:100;
        auc = zeros(numel(criteria), 2);
        
        % Calculate the percentage of firing rates greater than each criterion for both null and preferred distributions
        for c = 1:length(criteria)
            criterion = criteria(c);
            left_percentage = sum(leftward_choices > criterion) / numel(leftward_choices);
            right_percentage = sum(rightward_choices > criterion) / numel(rightward_choices);
            % both neurons are selective, so make ROC curve positive for
            % each neuron's preferred direction
            if j == 1 % left-preferring neuron
                auc(c, :) = [right_percentage, left_percentage];
            elseif j == 2 % right-preferring neuron
                auc(c, :) = [left_percentage, right_percentage];
            end
        end
    
        % Calculate the area under the curve (AUC) using trapz (aka perform
        % numerical integration) 
        AUC_values(i, j) = -trapz(auc(:, 1), auc(:, 2));

    end
end

% Generate histogram of AUC values
% figure; hold on;
% histogram(AUC_values(:, 1), 10); % Adjust the number of bins as needed, how do I know appropriate bins? 
% histogram(AUC_values(:, 2), 10); % Adjust the number of bins as needed, how do I know appropriate bins? 

% Generate histogram of AUC values
figure; hold on;
histogram(AUC_values(:, 1), 10, 'DisplayName', 'Neuron 1 (Left Preferring)'); 
histogram(AUC_values(:, 2), 10, 'DisplayName', 'Neuron 2 (Right Preferring)'); 
xlabel('Area Under the Curve (AUC)');
ylabel('Frequency');
title('Distribution of AUC Values for Left and Right Preferring Neurons');
legend('show');



% Calculate choice probability (mean AUC)
choice_probability = mean(AUC_values(:, 1));
disp(['Choice Probability Neuron 1: ', num2str(choice_probability)]);
choice_probability = mean(AUC_values(:, 2));
disp(['Choice Probability Neuron 2: ', num2str(choice_probability)]);
%%
% Define variables
num_coherences = 6; % Given in the assignment
num_neurons = 2; % Left motion and right motion neurons 
AUC_values_combined = zeros(num_coherences, 1);
total_left = zeros(num_coherences, 1); 
total_right = zeros(num_coherences, 1);

% For loop for each coherence
for i = 1:num_coherences
    total_left(i) = nnz(MTdata(:, 3, i) == 0);
    total_right(i) = nnz(MTdata(:, 3, i) == 1);
    rightward_choices = MTdata(MTdata(:, 3, i) == 1, :, i); % Assuming data structure is [coherence, neuron, choice, trial]
    leftward_choices = MTdata(MTdata(:, 3, i) == 0, :, i);
    
    % Slide criterion from 0 to 100
    criteria = 0:100;
    auc_combined = zeros(numel(criteria), 1);
    
    % Calculate the percentage of firing rates greater than each criterion for combined data
    for c = 1:length(criteria)
        criterion = criteria(c);
        left_percentage = sum(leftward_choices > criterion, 'all') / numel(leftward_choices);
        right_percentage = sum(rightward_choices > criterion, 'all') / numel(rightward_choices);
        auc_combined(c) = (right_percentage + left_percentage) / 2; % Average of percentages
    end
    
    % Calculate the area under the curve (AUC) using trapz (perform numerical integration) 
    AUC_values_combined(i) = trapz(criteria, auc_combined);
end

% Generate histogram of combined AUC values
figure; 
histogram(AUC_values_combined, 10, 'DisplayName', 'Combined Neurons'); 
xlabel('Area Under the Curve (AUC)');
ylabel('Frequency');
title('Distribution of Combined AUC Values');
legend('show');

% Calculate choice probability (mean AUC)
choice_probability_combined = mean(AUC_values_combined);
disp(['Choice Probability (Combined Neurons): ', num2str(choice_probability_combined)]);

%%
%%
%define Weibull function
p=@(b,x) 1-.5*exp(-(x/b(1)).^b(2));
%initialize values for alpha and beta
init=[10,2];

%fit using nonlinear regression
[b_fit1,~]=nlinfit(coherence*100,auc,p,init);

line_fit=1-.5*exp(-(linspace(1,70,100)/b_fit1(1)).^b_fit1(2));

fig3=figure(3);
semilogx(linspace(1,70,100),line_fit,'Color','r','LineWidth',1)
hold on
plot(coherence*100+1,auc,'o','MarkerEdgeColor','b')
xlabel('% Coherence')
ylabel('Area Under Curve')
title('Neurometric Curve')

legend({'Weibull Function','Data'},"Location","northwest")

exportgraphics(fig3,'fig3.png','Resolution',300)
