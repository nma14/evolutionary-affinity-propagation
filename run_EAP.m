% Example on how to run evolutionary affinity propagation with sample data
% sample_data.mat contains synthetic data generated from 2 Gaussian
% distributions that move closer from time steps 1-10, have some points
% break off into a third cluster at t=11 and t=12, and continue being drawn
% from the same distribution from t=13 to t=25.


load('sample_data.mat')
N=length(data{1});
T=length(data);

% Normalize data by subtracting global mean and dividing by global std. 
% Get similarity matrix using negative squared Euclidean distance as the
% similarity and set the preference values (diagonal) to the minimum
% similarity.

data_norm = cell(T);
S = zeros(N,N,T);
Sp = zeros(N,N,T);
pref = zeros(T,1);
datam=cell2mat(data);
mean_all = mean(datam);
std_all = std(datam);

for t=1:T
    data1 = data{t};
    data_norm{t}=(data1-mean_all)./repmat(std_all,size(data1,1),1);
    D=-pdist(data_norm{t}, 'euclidean').^2;
    S(:,:,t) = squareform(D);
    Sp(:,:,t) = S(:,:,t);
    for k=1:N
        S(k,k,t)=0;
    end
    Stmp = S(:,:,t);
    pref(t) = min(min(Stmp(Stmp~=0)));
    for k=1:N
        Sp(k,k,t)=pref(t);
    end
end
Spcell = reshape(mat2cell(Sp, N,N,ones(T,1)),T,1,1);


% Run evolutionary affinity propagation to obtain clustering.
maxiter=200;
conviter=20;
lam=0.9;
omega=1; 
gamma=2; 

cluster_id= evolutionary_affinity_propagation(Spcell, data_norm, lam, gamma, omega, maxiter, conviter);

