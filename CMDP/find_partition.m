function idxs = find_partition(sampleContext, centroids)
[m, n] = size(centroids);
distances = [];
for i = 1:m
    distances = [distances, norm(sampleContext-sampleContext, 2)];
end
idxs = find(distances==max(distances));
if length(idxs) >1
    idxs = idxs(randi([1 length(idxs)],1));
end
end