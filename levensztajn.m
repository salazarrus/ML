function [dist] = levensztajn(string1, string2)
  
m = size(string1,2);

n = size(string2,2);

cost = zeros(m+1,n+1);
operations = zeros(m+1,n+1);


cost(:,1) = 0:1:m ;

cost(1,:) = 0:1:n ;

for i=2:m+1,
  for j=2:n+1,
    string1(1,i-1);
    string2(1,j-1);
    unitCost = 2*(1 -(string1(1,i-1) == string2(1,j-1)));
    
    insertion = (cost(i,j-1)+1); 
    
    deletion = (cost(i-1,j)+1) ;
    
    substitution= (cost(i-1,j-1)+ unitCost);
    
    cost(i,j) = insertion;
    operation = 0;
    
    if cost(i,j) > deletion,
      cost(i,j) = deletion;
      operation = 1;
    end;
    
    if cost(i,j) > substitution,
      cost(i,j) = substitution;
      operation = 2;
    end;
    
    operations(i,j) = operation;
    
    end;
end;


cost
operations
dist = cost(m+1,n+1);