% Use relative path from script location to matrices directory in artifact
matricesDir = fullfile('..', '..', 'matrices');
logsDir = fullfile('..', '..', 'logs');
removal_percentages = [0.01, 0.05, 0.1];

% Create logs directory if it doesn't exist
if ~exist(logsDir, 'dir')
    mkdir(logsDir);
end

% Open CSV log files for writing (in logs directory).
csvFile_inf_norm = fullfile(logsDir, 'inf_norm_os.csv');
csvFile_norm2_os = fullfile(logsDir, 'norm2_os.csv');
csvFile_norm2_o = fullfile(logsDir, 'norm2_o.csv');
csvFile_norm2_s = fullfile(logsDir, 'norm2_s.csv');
csvFile_diag_min = fullfile(logsDir, 'diag_min.csv');

fid_inf_norm = fopen(csvFile_inf_norm, 'w');
fid_norm2_os = fopen(csvFile_norm2_os, 'w');
fid_norm2_o = fopen(csvFile_norm2_o, 'w');
fid_norm2_s = fopen(csvFile_norm2_s, 'w');
fid_diag_min = fopen(csvFile_diag_min, 'w');

% Write CSV headers
fprintf(fid_inf_norm, 'Matrix Name,Sparsification Ratio,Infinity Norm\n');
fprintf(fid_norm2_os, 'Matrix Name,Sparsification Ratio,2-Norm of Aos\n');
fprintf(fid_norm2_o, 'Matrix Name,2-Norm\n');
fprintf(fid_norm2_s, 'Matrix Name,Sparsification Ratio,2-Norm of E\n');
fprintf(fid_diag_min, 'Matrix Name,Sparsification Ratio,Smallest Diagonal Entry (Abs)\n');

% Open a log file for skipped runs (in logs directory).
skippedFile = fullfile(logsDir, 'skipped_matrices.txt');
fid_skipped = fopen(skippedFile, 'w');

% Get all subfolders in the matrices directory.
folders = dir(matricesDir);
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name}, {'.', '..'}));

% ========== Optional Bookmark Functionality ==========
bookmark = '';  % Set to '' to process all folders
[~, sortIdx] = sort({folders.name});
folders = folders(sortIdx);

if ~isempty(bookmark)
    idx = find(strcmp({folders.name}, bookmark), 1);
    if isempty(idx)
        fprintf('Bookmark "%s" not found. Processing all folders.\n', bookmark);
    else
        fprintf('Bookmark "%s" found. Skipping folders before this one.\n', bookmark);
        folders = folders(idx:end);
    end
end
% =====================================================

%% Process each matrix folder.
for f = 1:length(folders)
    folderName = folders(f).name;
    folderPath = fullfile(matricesDir, folderName);
    fprintf('\nProcessing matrix: %s\n', folderName);

    % Original file: e.g., "Kuu/Kuu.mtx"
    original_file = fullfile(folderPath, [folderName, '.mtx']);
    if ~exist(original_file, 'file')
        fprintf(fid_skipped, 'Folder %s: missing original file: %s\n', folderName, original_file);
        fprintf('Skipping folder %s (missing original file).\n', folderName);
        continue;
    end

    try
        Ao = mmread(original_file);
    catch ME
        fprintf(fid_skipped, 'Folder %s: error loading original file: %s\n', folderName, ME.message);
        fprintf('Skipping folder %s (error loading original file).\n', folderName);
        continue;
    end

    % --- Compute norms of Ao ---
    try
        inf_norm_Ao = norm(Ao, inf);

        lambda_max_o = eigs(Ao, 1, 'la');
        norm2_Ao = abs(lambda_max_o);
    catch ME
        fprintf(fid_skipped, 'Folder %s: error computing norms of Ao: %s\n', folderName, ME.message);
        fprintf('Skipping folder %s (error computing norms of Ao).\n', folderName);
        continue;
    end

    % --- Log Ao metrics ---
    fprintf(fid_inf_norm, '%s,%.2f,%g\n', folderName, 0.0, inf_norm_Ao);
    fprintf(fid_norm2_o, '%s,%g\n', folderName, norm2_Ao);
    fprintf('Logged original matrix: %s\n', folderName);

    for p = 1:length(removal_percentages)
        perc = removal_percentages(p);
        percStr = num2str(perc, '%.2f');
        sparsified_file = fullfile(folderPath, [folderName, '_', percStr, '.mtx']);

        if ~exist(sparsified_file, 'file')
            fprintf(fid_skipped, 'Folder %s, removal ratio %s: missing sparsified matrix.\n', folderName, percStr);
            fprintf('Skipping run for folder %s, removal ratio %s (missing file).\n', folderName, percStr);
            continue;
        end

        try
            Aos = mmread(sparsified_file);
        catch ME
            fprintf(fid_skipped, 'Folder %s, removal ratio %s: error loading Aos: %s\n', folderName, percStr, ME.message);
            fprintf('Skipping run for folder %s, removal ratio %s (load error).\n', folderName, percStr);
            continue;
        end

        % --- Load perturbation matrix s ---
        perturbation_file = fullfile(folderPath, [folderName, '_', percStr, '_ptb.mtx']);
        if ~exist(perturbation_file, 'file')
            fprintf(fid_skipped, 'Folder %s, removal ratio %s: missing perturbation matrix.\n', folderName, percStr);
            fprintf('Skipping run for folder %s, removal ratio %s (missing perturbation file).\n', folderName, percStr);
            continue;
        end

        try
            s = mmread(perturbation_file);
        catch ME
            fprintf(fid_skipped, 'Folder %s, removal ratio %s: error loading s: %s\n', folderName, percStr, ME.message);
            fprintf('Skipping run for folder %s, removal ratio %s (perturbation load error).\n', folderName, percStr);
            continue;
        end

        % --- Compute all norms and metrics ---
        try
            inf_norm_Aos = norm(Aos, inf);

            lambda_max_os = eigs(Aos, 1, 'la');
            norm2_Aos = abs(lambda_max_os);

            lambda_max_s = eigs(s, 1, 'la');
            norm2_s = abs(lambda_max_s);
            
            % Compute smallest diagonal entry (absolute value) of Aos
            diag_Aos = diag(Aos);
            min_diag_Aos = full(min(abs(diag_Aos)));
        catch ME
            fprintf(fid_skipped, 'Folder %s, removal ratio %s: error computing norms: %s\n', folderName, percStr, ME.message);
            fprintf('Skipping run for folder %s, removal ratio %s (norm computation error).\n', folderName, percStr);
            continue;
        end

        % --- Log all metrics ---
        fprintf(fid_inf_norm, '%s,%.2f,%g\n', folderName, perc, inf_norm_Aos);
        fprintf(fid_norm2_os, '%s,%.2f,%g\n', folderName, perc, norm2_Aos);
        fprintf(fid_norm2_s, '%s,%.2f,%g\n', folderName, perc, norm2_s);
        fprintf(fid_diag_min, '%s,%.2f,%g\n', folderName, perc, min_diag_Aos);
        fprintf('Logged run: %s, removal ratio %s.\n', folderName, percStr);
    end
end

% Close the log files.
fclose(fid_inf_norm);
fclose(fid_norm2_os);
fclose(fid_norm2_o);
fclose(fid_norm2_s);
fclose(fid_diag_min);
fclose(fid_skipped);

fprintf('\nAll processing completed.\n');
fprintf('CSV files written:\n');
fprintf('  - Infinity norm (Aos): %s\n', csvFile_inf_norm);
fprintf('  - 2-norm (Aos): %s\n', csvFile_norm2_os);
fprintf('  - 2-norm (Ao): %s\n', csvFile_norm2_o);
fprintf('  - 2-norm (s): %s\n', csvFile_norm2_s);
fprintf('  - Smallest diagonal entry (Aos): %s\n', csvFile_diag_min);
fprintf('Skipped runs logged in: %s\n', skippedFile);
