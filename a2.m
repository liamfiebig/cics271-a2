function a2

% Function for CISC271, Winter 2020, Assignment #2

% % Read the test data from a CSV file
wine = 'wine.csv';
M = csvread(wine,1,1); % M is the independent data vectors

% Extract the data matrix and the Y cluster identifiers
Xmat = transpose(M); %formatting datavectors into Matlab form
yvec = transpose([[ones(1,59)] [2*ones(1,71)] [3*ones(1,48)]]); % yvec is geneterated from data in file

% Problem size
[m , n] = size(Xmat);

% Compute the pair of columns of Xmat with the lowest DB index
% loops though the list n * n times to find the two columns that have the
% lowest dbindex
lowest = 1000001;
ndx_lowest = [];
for ii=1:n
    for jj=1:n
        if ii ~= jj 
            ndx_lo = [ii jj];
            val_lo = dbindex(Xmat(:, ndx_lo), yvec);
            if val_lo < lowest
                lowest = val_lo;
                ndx_lowest = ndx_lo;
            end
        end     
    end  
end
ndx_lo = ndx_lowest;
val_lo = lowest;

% Compute the PCA's of the data using the SVD; score the clusterings
% PMAT contains the PCA score of the original data
% ZMAT contains the PCA score of the standardized data

zmXmat = Xmat;
% piecewise creation of zero-mean matrix
for ii = 1:n
   zmXmat(:,ii) = (zmXmat(:,ii) - mean(zmXmat(:,ii))); 
end
[U, S, V] = svd(zmXmat);
%creation of a zero mean scoring matrix from zero mean matrix and scoring
%columns of the V component.
pmat = zmXmat*V(:, [1 2]);

% standardizing the zero-mean matrix
standard = zscore(zmXmat);
[U, S, V] = svd(standard);
%creation of a standard scoring matrix from standard matrix and scoring
%columns of the V component.
zmat = standard*V(:, [1 2]);

val_p = dbindex(pmat, yvec);
val_z = dbindex(zmat, yvec);

% Display the cluster scores and the indexes; plot the data

disp(sprintf('For X data: score = %0.4f   indexes = [%d,%d]', ...
    val_lo, ndx_lo));
disp(sprintf('For P data: score = %0.4f', ...
    val_p));
disp(sprintf('For Z data: score = %0.4f', ...
    val_z));
figure(1);
gscatter(Xmat(:, ndx_lo(1)), Xmat(:,  ndx_lo(2)), yvec);
figure(2);
gscatter(pmat(:, 1), pmat(:, 2), yvec);
figure(3);
gscatter(zmat(:, 1), zmat(:, 2), yvec);

end

function score = dbindex(Xmat, yvec)
% SCORE=DBINDEX(XMAT,YVEC) computes the Davies-Bouldin index
% for a design matrix XMAT by using the values in YVEC as labels.
% The calculation is performed by the EVALCLUSTERS function .
%
% INPUTS:
%        XMAT  - MxN design matrix, each row is an observation and
%                each column is a variable
%        YVEC  - Mx1 label vector, each entry is an observation label
% OUTPUT:
%        SCORE - non-negative scalar, smaller is "better"

% Compute the structure for cluster evaluation
caStruct = evalclusters(Xmat, yvec, 'DaviesBouldin');

% The DB index score is the first "criterion value"
score = caStruct.CriterionValues(1);
end