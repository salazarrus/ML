function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

bigDelta1 = zeros(size(Theta1));

bigDelta2 = zeros(size(Theta2));

for i=1:m,
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                 FORWARD-PROP                %   
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  x = X(i,:); %1x(input_layer_size)
  
  actualLabel = y(i,:); %scalar
  
  % generate single-one network
  y_vect = zeros(num_labels,1); %num_labelsx1
  
  y_vect(actualLabel) = 1;  

  a1 = [1; x(:)]; % (input_layer_size+1)x1
  
  z1 = Theta1 * a1;
  
  a2 = sigmoid(z1); % (hidden_layer_size)x1
  
  a2 = [1; a2]; %(hidden_layer_size+1)x1
  
  z2 = Theta2 * a2;
  
  a3 = sigmoid(z2); % (num_labels)x 1
  
  singleCost = -y_vect' * log(a3) - (1-y_vect')*log(1-a3);
  
  J = J + singleCost;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                  BACK-PROP                  %   
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  delta3 = a3-y_vect; % num_labelsx1
  
  delta2 = Theta2' * (delta3 .* sigmoidGradient(z2)) ; %(hidden_layer_size+1)x1
  
  bigDelta2 = bigDelta2 + delta3(2:end) * (a2');
  
  bigDelta1 = bigDelta1 + delta2(2:end) * (a1');
   
  
end;

J = J/m;

Theta2_grad = bigDelta2/m;
  
Theta1_grad = bigDelta1/m;

theta2zeroColumn = Theta2;

theta2zeroColumn(:,1) = zeros(num_labels,1);

theta1zeroColumn = Theta1;

theta1zeroColumn(:,1) = zeros(hidden_layer_size,1);



Theta2_grad = Theta2_grad + (lambda/m)*theta2zeroColumn;

Theta1_grad = Theta1_grad + (lambda/m)*theta1zeroColumn;




regularization1 = sum(sum(Theta1(:, 2:end).^2));
 
regularization2 = sum(sum(Theta2(:, 2:end).^2)); 

regularization = (lambda* (regularization1 + regularization2))/(2*m);

J = J + regularization;
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
