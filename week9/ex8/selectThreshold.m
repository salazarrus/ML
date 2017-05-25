function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;
pval;
yval;
stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
        
    predictions = pval < epsilon;
    
    dif = predictions - yval;
    
    summed = predictions + yval;
    
    true_positives = sum(summed==2);
    
    false_positives =sum(dif==1); 
    
    false_negatives =sum(dif==-1);
    
    if true_positives + false_positives == 0,
      precision = 0;
    else 
      precision = true_positives / (true_positives + false_positives);
    end;
    
    if true_positives + false_negatives == 0,
      recall = 0;
    else
      recall = true_positives / (true_positives + false_negatives); 
    end;
   
    if precision + recall == 0,
      F1 = 0;
    else
      F1 = (2*precision*recall)/ (precision + recall);  
    end;
    
    if F1 > bestF1,
      bestF1 = F1;
      bestEpsilon = epsilon;  
    end;
    
    
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions


end

end