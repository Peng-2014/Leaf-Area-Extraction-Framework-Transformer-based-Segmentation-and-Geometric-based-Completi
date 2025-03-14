clc
clear
ye=importdata('point.txt');

%% Downsampling
li=0.5; % Adjustment coefficient [0~1]
PointCloud = ye(:,1:3)*500*li; 
samplingRate = 0.2; %Sampling ratio[0~1]


% Calculate number of points to keep
numPointsToKeep = round(size(PointCloud, 1) * samplingRate);

% Random downsampling
originalPointCloud = datasample(PointCloud, numPointsToKeep, 'Replace', false);

%% Find the farthest points
pointCloudData = originalPointCloud; % 3D point cloud data

% Create pointCloud object
ptCloud = pointCloud(pointCloudData);

% Calculate pairwise distances
distances = pdist(ptCloud.Location);

% Convert to square matrix
distanceMatrix = squareform(distances);

% Find maximum distance and indices
[maxDistance, maxIndex] = max(distanceMatrix(:));

% Convert linear index to subscripts
[row, col] = ind2sub(size(distanceMatrix), maxIndex);

% Get farthest points coordinates
farthestPoints = [ptCloud.Location(row, :); ptCloud.Location(col, :)];

disp('Coordinates of the two farthest points:');
disp(farthestPoints);
disp(['Distance between them: ', num2str(maxDistance)]);

% Create Z-axis from farthest points
zAxis = farthestPoints(2, :) - farthestPoints(1, :);
zAxis = zAxis / norm(zAxis);

% Calculate rotation matrix
rotationMatrix = vrrotvec2mat(vrrotvec([0 0 1], zAxis));
transformedPointCloud=ptCloud.Location*rotationMatrix;
% Translation
translationVector=-transformedPointCloud(row, :);
transformedPointCloud=transformedPointCloud+translationVector;

farPoints = [transformedPointCloud(row, :); transformedPointCloud(col, :)];
disp('Transformed coordinates of farthest points:');
disp(farPoints);

%% Project to 2D plane and find highest points
xuanzhuan=100; % Set rotation ratio
[m,d]=xztoumian(transformedPointCloud);
areachushi=m(1);
l=1;
for i=1:xuanzhuan
    angle=(i*pi)/xuanzhuan;
rotxing=[cos(angle),sin(angle),0;-sin(angle),cos(angle),0;0,0,1];
touyingpointcloud=transformedPointCloud*rotxing;

% Project to XZ plane
[m,d]=xztoumian(touyingpointcloud);
area=m(1);
if  areachushi<=area
    areachushi=area;
     l=i;
end

end
angle=(l*pi)/xuanzhuan;
rotxing=[cos(angle),sin(angle),0;-sin(angle),cos(angle),0;0,0,1];
touyingpointcloud=transformedPointCloud*rotxing;
dd=tisuhuashangdian(touyingpointcloud,1,2); % Voxel sampling

[m,d]=xztoumian(touyingpointcloud);
areachushi=m(1);

%% Delaunay Triangulation
points = dd; 
xd=max(points(:,1));
xx=min(points(:,1));
yd=max(points(:,3));
yx=min(points(:,3));
zx=min(points(:,2));
[x,y] = meshgrid((xx-xx+1):(xd-xx+1),(yx-yx+1):(yd-yx+1));
points(:,1)=points(:,1)-xx+1;
points(:,3)=points(:,3)-yx+1;
points(:,2)=points(:,2)-zx+1;

z=zeros((yd-yx+1),(xd-xx+1));
ww=size(points);
for i=1:ww(1)
    z(points(i,3),points(i,1))=points(i,2);
end

T = delaunay(x,y);

t=0;
for i = 1:size(T, 1)
    % Get triangle vertices
    if z(T(i, 1))==0||z(T(i, 2))==0||z(T(i, 3))==0
    else
         t=t+1;
     S(t,:)=T(i, :);
    end
 end

figure;
h = trimesh(S, -x, y, z);
set(h, 'FaceColor', 'w');    
set(h, 'EdgeColor', 'k');    
xlabel('X');
ylabel('Y');
zlabel('Z');
axis equal;
axis off;

totalArea = 0;

for i = 1:size(S, 1)
    % Calculate triangle area
    P1 = [x(S(i, 1));y(S(i, 1));z(S(i, 1))];
    P2 = [x(S(i, 2));y(S(i, 2));z(S(i, 2))];
    P3 = [x(S(i, 3));y(S(i, 3));z(S(i, 3))];
    
    a = norm(P2 - P1);
    b = norm(P3 - P2);
    c = norm(P1 - P3);
    
    s = (a + b + c) / 2;
    
    totalArea = totalArea + sqrt(s * (s - a) * (s - b) * (s - c));
end
totalArea= totalArea/((li*15.5)*(li*15.5));
disp(['Total area of Delaunay triangulation: ', num2str(totalArea)]);
zz=max(max(z));

%% Create Image
d=points;
pixelCoordinates = points;
xmin=min(d(:,1));
xmax=max(d(:,1));
ymin=min(d(:,3));
ymax=max(d(:,3));

pixelCoordinates(:,1)=d(:,1)-xmin+100;
pixelCoordinates(:,3)=d(:,3)-ymin+100;

mm=xmax-xmin+200;
nn=ymax-ymin+200;

% Create black image
imageSize = [nn, mm]; 
image = zeros(imageSize);
for i = 1:size(pixelCoordinates, 1)
    x = pixelCoordinates(i, 1);
    y = pixelCoordinates(i, 3);
    z = pixelCoordinates(i, 2)./zz;
    image(y, x) = z; 
end

%% body Completion

kernelType = 'square';
kernelSize = 10;
morphKernel = strel(kernelType, kernelSize);

if kernelSize > 20
    error('Kernel dimension overflow');
else
    configValid = true;
end

sourceImage = image;
interimImage = sourceImage;
processedImage = interimImage;

preIterations = 3;
for warmup = 1:preIterations
    cycleTracker = warmup;
end

enhancedImage = imdilate(processedImage, morphKernel);

pixelMetrics = [sum(enhancedImage(:)), numel(enhancedImage)];
processedImage = enhancedImage;

imageBackup = processedImage; 



refinedImage = imerode(processedImage, morphKernel);

assert(all(size(refinedImage) == size(sourceImage)), 'Dimension mismatch');

resultArchive = struct('phase1', enhancedImage, 'phase2', refinedImage);

%% Map back to 3D
z=refinedImage.*zz;
[m,n]=size(z);
[x,y] = meshgrid(0:n-1,0:m-1);

T = delaunay(x,y);

t=0;
for i = 1:size(T, 1)
    if z(T(i, 1))==0||z(T(i, 2))==0||z(T(i, 3))==0
    else
         t=t+1;
     S(t,:)=T(i, :);
    end
 end
figure;

h = trimesh(S, -x, y, z);
set(h, 'FaceColor', 'w');    
set(h, 'EdgeColor', 'k');    
xlabel('X');
ylabel('Y');
zlabel('Z');
axis equal;
axis off;

totalArea = 0;

for i = 1:size(S, 1)
    P1 = [x(S(i, 1));y(S(i, 1));z(S(i, 1))];
    P2 = [x(S(i, 2));y(S(i, 2));z(S(i, 2))];
    P3 = [x(S(i, 3));y(S(i, 3));z(S(i, 3))];
    
    a = norm(P2 - P1);
    b = norm(P3 - P2);
    c = norm(P1 - P3);
    
    s = (a + b + c) / 2;
    
    totalArea = totalArea + sqrt(s * (s - a) * (s - b) * (s - c));
end
totalArea=totalArea/((li*15)*(li*15));
disp(['Total area after completion: ', num2str(totalArea)]);