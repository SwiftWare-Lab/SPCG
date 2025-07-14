% This script loops through subfolders in a base directory.
% In each folder, it finds the Matrix Market file named <foldername>.mtx,
% then for each removal percentage it sparsifies the off-diagonal entries
% (i.e., sets to zero if abs(value) <= threshold) and computes the perturbation E,
% such that A = A_sparsified + E. Both A_sparsified and E are then saved.

baseDir = fullfile('..', '..', 'matrices');

% Set the bookmark folder name.
% To process all folders, set bookmark = '';
bookmark = '';

% List all items in the base directory.
items = dir(baseDir);

% Filter for directories (excluding '.' and '..')
items = items([items.isdir]);
items = items(~ismember({items.name}, {'.', '..'}));

% Sort folder names alphabetically to ensure a fixed order.
[~, sortIdx] = sort({items.name});
items = items(sortIdx);

% If a bookmark is set, skip folders until the bookmark is reached.
if ~isempty(bookmark)
    idx = find(strcmp({items.name}, bookmark), 1);
    if isempty(idx)
        fprintf('Bookmark "%s" not found. Processing all folders.\n', bookmark);
    else
        fprintf('Bookmark "%s" found. Skipping folders before this one.\n', bookmark);
        items = items(idx:end);
    end
end

% Define the removal percentages to use.
removal_percentages = [0.01, 0.05, 0.1];

for k = 1:length(items)
    % Process only directories that are not '.' or '..'
    if items(k).isdir && ~ismember(items(k).name, {'.', '..'})
        folderName = items(k).name;
        folderPath = fullfile(baseDir, folderName);
        % Construct the matrix file name (e.g., Kuu.mtx inside folder 'Kuu')
        matrixFile = fullfile(folderPath, [folderName, '.mtx']);

        if exist(matrixFile, 'file')
            fprintf('Processing matrix file: %s\n', matrixFile);
            % Read the sparse matrix using mmread (assumed to be in your path)
            try
                A = mmread(matrixFile);
            catch ME
                fprintf('Error reading matrix file %s: %s\n', matrixFile, ME.message);
                continue;  % Skip to the next matrix folder
            end

            % Get the size of A.
            [m, n] = size(A);
            if m ~= n
                warning('Matrix %s is not square. Skipping...', folderName);
                continue;
            end

            % Get the nonzero entries of A.
            [I, J, V] = find(A);

            % Sort the absolute values of all nonzero entries.
            sortedVals = sort(abs(V));
            numVals = numel(sortedVals);

            for p = 1:length(removal_percentages)
                perc = removal_percentages(p);
                % Determine the index (make sure it is at least 1)
                idx = max(1, floor(numVals * perc));
                removal_threshold = sortedVals(idx);
                fprintf('  Removal percentage %.2f: threshold = %g\n', perc, removal_threshold);

                % Identify off-diagonal entries (I ~= J) with abs(value) <= removal_threshold.
                mask = (I ~= J) & (abs(V) <= removal_threshold);

                % Create perturbation matrix E from these entries.
                E = sparse(I(mask), J(mask), V(mask), m, n);

                % Compute the sparsified matrix as A_sparsified = A - E.
                A_sparsified = A - E;

                % Construct output file names.
                percStr = num2str(perc, '%.2f');
                outFile_spars = fullfile(folderPath, [folderName, '_', percStr, '.mtx']);
                outFile_ptb   = fullfile(folderPath, [folderName, '_', percStr, '_ptb.mtx']);

                % Save the matrices using mmwrite.
                mmwrite(outFile_spars, A_sparsified);
                mmwrite(outFile_ptb, E);
                fprintf('    Exported %s and %s\n', outFile_spars, outFile_ptb);
            end
        else
            fprintf('Matrix file %s not found in folder %s.\n', [folderName, '.mtx'], folderName);
        end
    end
end

fprintf('Processing completed.\n');
