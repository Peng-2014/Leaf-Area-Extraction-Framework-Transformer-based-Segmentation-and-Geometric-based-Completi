function [m,ss]=xztoumian(x)
pointcloud=x;
d=round([pointcloud(:,1),pointcloud(:,3)]);
uniqueDataPairs = unique(d, 'rows');
m=size(uniqueDataPairs);
ss=uniqueDataPairs;
end
