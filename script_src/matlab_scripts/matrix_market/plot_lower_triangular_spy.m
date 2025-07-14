function plot_lower_triangular_spy(filename)
    data = readmatrix(filename, 'FileType', 'text');

    if size(data,2) < 3
        error('Input file must be in coordinate format with at least 3 columns (i, j, val).');
    end

    % Fix zero-based indexing by shifting all indices by +1
    rows = int32(data(:,1)) + 1;
    cols = int32(data(:,2)) + 1;
    vals = data(:,3);
    n = max(max(rows), max(cols));
    A = sparse(rows, cols, vals, n, n);

    % Extract lower triangular part
    L = tril(A);

    % Create invisible figure for headless mode
    fig = figure('Visible', 'off');
    axes('Parent', fig); % Ensure axes exist
    spy(L);
    title('Spy Plot of Lower Triangular Matrix', 'FontWeight', 'bold', 'FontSize', 14);
    axis tight;

    % Set renderer to painters for vector compatibility in headless mode
    set(fig, 'Renderer', 'painters');

    % Define output filenames
    [folder, name, ~] = fileparts(filename);
    pdf_name = fullfile(folder, [name '_lower_triangular_spy.pdf']);
    png_name = fullfile(folder, [name '_lower_triangular_spy.png']);

    % Save plots
    saveas(fig, pdf_name);
    saveas(fig, png_name);

    fprintf("Spy plot saved to:\n  - %s\n  - %s\n", pdf_name, png_name);

    close(fig); % Clean up figure
end
