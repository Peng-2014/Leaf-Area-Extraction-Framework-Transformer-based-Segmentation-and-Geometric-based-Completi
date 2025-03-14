function [ss] = tisuhuashangdian(x, y, z)
% x: Input point cloud
% y: Voxel size
% z: Axis configuration mode (1,2,3 for different axis orders)
% ss: Voxelized point cloud with highest points

% Create point cloud object and round coordinates
PointCloud = pointCloud(round(x));
voxelSize = y; % Set voxel grid size

% Perform voxel grid downsampling
voxelizedPointCloud = pcdownsample(PointCloud, 'gridAverage', voxelSize);

points = voxelizedPointCloud.Location;
cloud = points;

% Reorder axes based on configuration parameter z
if z == 3
    % Original XYZ order
    x_values = [cloud(:, 1)];
    y_values = [cloud(:, 2)];
    z_values = [cloud(:, 3)];
elseif z == 1
    % ZYX -> XYZ reordering
    z_values = [cloud(:, 1)];
    y_values = [cloud(:, 2)];
    x_values = [cloud(:, 3)];
elseif z == 2
    % XZY -> XYZ reordering
    x_values = [cloud(:, 1)];
    z_values = [cloud(:, 2)];
    y_values = [cloud(:, 3)];
end

% Find unique XY combinations
unique_xy = unique([x_values, y_values], 'rows');

% Initialize result container
result_points = zeros(size(unique_xy, 1), 3);

% Process each unique XY coordinate
for i = 1:size(unique_xy, 1)
    current_xy = unique_xy(i, :);
    
    % Find indices matching current XY
    matching_indices = find(x_values == current_xy(1) & y_values == current_xy(2));
    
    % Find point with maximum Z value
    [~, max_z_index] = max(z_values(matching_indices));
    
    % Store highest point
    result_points(i, :) = cloud(matching_indices(max_z_index), :);
end

ss = result_points;

end