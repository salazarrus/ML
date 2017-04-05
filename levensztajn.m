function [dist] = levensztajn(string1, string2)
  
m = size(string1,2);

n = size(string2,2);

cost = zeros(m+1,n+1);

cost(:,1) = 0:1:m ;

cost(1,:) = 0:1:n ;

for i=2:m+1,
  for j=2:n+1,
    string1(1,i-1);
    string2(1,j-1);
    unitCost = 1 -(string1(1,i-1) == string2(1,j-1));
    cost(i,j) = min([(cost(i,j-1)+1) (cost(i-1,j)+1) (cost(i-1,j-1)+ unitCost)]);  
    end;
end;


cost
dist = cost(m+1,n+1);