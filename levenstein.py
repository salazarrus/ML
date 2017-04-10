x = "abcd";
y = "bfde";

print(x[0])

m=len(x)
n=len(y)

distances=[]

for r in range(0,m+1):
	row = [0]*(n+1);
	row[0] = r;
	distances.append(row)


for c in range(0,n+1):
	distances[0][c]=c


print(distances);


for r in range(1,m+1) :
	for c in range(1,n+1):
		cost = 0;
		if x[r-1]!=y[c-1]:
			cost=2        
		distances[r][c] = min(distances[r-1][c]+1,distances[r][c-1]+1, distances[r-1][c-1]+cost);

		
print(distances[m][n])

