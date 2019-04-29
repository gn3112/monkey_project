function [RMSE, accuracy] = valid_func(testData, modelParameters)
meanSqError = 0;
n_predictions = 0;
accuracy = 0; 
for tr=1:size(testData,1)
    %     display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    for direc=randperm(8)
        decodedHandPos = [];
        
        times=320:20:size(testData(tr,direc).spikes,2);
        
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
            accuracy = accuracy_class + accuracy;
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            
        end
        n_predictions = n_predictions+length(times);
        %         hold on
        %         plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        %         plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    end
end

% legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions)
accuracy = accuracy/n_predictions
% rmpath(genpath(teamName))

end
