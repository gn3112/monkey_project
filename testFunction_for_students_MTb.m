% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 
function [RMSE] = testFunction_for_students_MTb()
tic
load('monkeydata_training.mat');
% Set random number generator
rng(2004);%2003 2004 2005 2006 2013
ix = randperm(length(trial));

% addpath(teamName);

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:80),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  

figure
hold on
axis square
grid

% Train Model
modelParameters = positionEstimatorTraining(trainingData);
accuracy = 0;
prevX = 0;
count = 0;
accuracy_direc = zeros(1,8);
meanSqError_direc = zeros(1,8);
for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
        decodedHandPos = [];

        times=320:20:size(testData(tr,direc).spikes,2);
%         X = zeros(1,length(times));
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY,accuracy_class] = positionEstimator(past_current_trial, modelParameters,direc);
            end
            accuracy_direc(1,direc) = accuracy_direc(1,direc) + accuracy_class; 
            accuracy = accuracy_class + accuracy;
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            meanSqError_direc(1,direc) = meanSqError_direc(1,direc) + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            if tr==1 && direc == 2
               disp('something')
               X(1,(t-300)/20) = prevX - decodedPosX;
               prevX = X(1,(t-300)/20);
               time_x = times;
               trial_x = testData(1,1);
            end
        end
        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
        
    end
    figure
    plot(time_x,X)
    raster_plot(trial_x,time_x)
end

legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions)
RMSE_direc = sqrt(meanSqError_direc/(n_predictions/8))
accuracy_direc = accuracy_direc./(n_predictions/8)
accuracy = accuracy/n_predictions
% rmpath(genpath(teamName))
toc
end
