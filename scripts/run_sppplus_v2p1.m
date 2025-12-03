% Run MATLAB SPPplus v2.1 on golden_data inputs.
% Usage inside MATLAB: run('scripts/run_sppplus_v2p1.m')

% Use script directory for portability
scriptpath = mfilename('fullpath');
scriptdir = fileparts(scriptpath);
basedir = fullfile(scriptdir, 'golden_data');
inpdir = fullfile(basedir, 'input');
outdir = fullfile(basedir, 'outputs', 'matlab');
if ~exist(outdir, 'dir'); mkdir(outdir); end

datasets = {'sin_fundamental', 'sin_noisy'};

addpath('/Users/b80985/Documents/MATLAB/SPPplus_v2p1');

% Save original working directory and change to script dir for SPPplus outputs
orig_dir = pwd;
cd(scriptdir);

for i=1:length(datasets)
    ds = datasets{i};
    infile = fullfile(inpdir, strcat(ds, '.csv'));

    % Read CSV and convert to tab-delimited .txt without header for SPPplus
    % SPPplus ftype=1 expects: whitespace-delimited .txt, no header
    data = readmatrix(infile);  % MATLAB R2019a+, auto-skips header
    txtfile = fullfile(scriptdir, strcat(ds, '.txt'));
    % Use dlmwrite with tab delimiter for reliable parsing by textscan
    dlmwrite(txtfile, data, 'delimiter', '\t', 'precision', '%.10f');

    % Configure RunSPPplus inputs programmatically
    fname = ds;  % SPPplus expects filename without extension
    ftype = 1;   % 1 = .txt file (space-delimited, no header)
    var_loc = [1,2,0,3]; % Time=col1, Strain=col2, Rate=0 (infer), Stress=col3
    var_conv = [1,1,1,1];
    data_trunc = [0,0,0];
    an_use = [1,1];
    omega = 2*pi; % matches generator
    % p=3 because input data has 3 cycles (from gen_inputs.py n_cycles=3)
    M = 39; p = 3; k = 8; num_mode = 2;
    out_type = 1; is_fsf = 1; save_figs = 0;

    % Call SPPplus_fourier_v2 and numerical
    [time_wave,resp_wave,L,fname_t] = SPPplus_read_v2(fname,ftype,var_loc,var_conv,data_trunc);
    out_set=[out_type,is_fsf,save_figs];
    SPPplus_fourier_v2(time_wave,resp_wave,L,omega,M,p,out_set,fname_t);
    SPPplus_numerical_v2(time_wave,resp_wave,L,omega,k,num_mode,out_set,fname_t);

    % Move outputs into golden_data/outputs/matlab
    movefile(strcat(fname_t,'_SPP_FOURIER.txt'), fullfile(outdir, strcat(ds,'_spp_data_out_fourier.txt')));
    movefile(strcat(fname_t,'_SPP_FOURIER_FSFRAME.txt'), fullfile(outdir, strcat(ds,'_fsf_data_out_fourier.txt')));
    if exist(strcat(fname_t,'_SPP_FOURIER_PLOT.jpg'), 'file')
        movefile(strcat(fname_t,'_SPP_FOURIER_PLOT.jpg'), fullfile(outdir, strcat(ds,'_plot_fourier.jpg')));
    end
    movefile(strcat(fname_t,'_SPP_NUMERICAL.txt'), fullfile(outdir, strcat(ds,'_spp_data_out_numerical.txt')));
    movefile(strcat(fname_t,'_SPP_NUMERICAL_FSFRAME.txt'), fullfile(outdir, strcat(ds,'_fsf_data_out_numerical.txt')));

    % Clean up converted .txt file
    delete(txtfile);
end

% Restore original working directory
cd(orig_dir);

disp('MATLAB SPPplus runs complete. Outputs in golden_data/outputs/matlab');
