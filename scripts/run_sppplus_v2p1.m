% Run MATLAB SPPplus v2.1 on golden_data inputs.
% Usage inside MATLAB: run('scripts/run_sppplus_v2p1.m')

basedir = fullfile(pwd, 'golden_data');
inpdir = fullfile(basedir, 'input');
outdir = fullfile(basedir, 'outputs', 'matlab');
if ~exist(outdir, 'dir'); mkdir(outdir); end

datasets = {'sin_fundamental', 'sin_noisy'};

addpath('/Users/b80985/Documents/MATLAB/SPPplus_v2p1');

for i=1:length(datasets)
    ds = datasets{i};
    infile = fullfile(inpdir, strcat(ds, '.csv'));
    % Configure RunSPPplus inputs programmatically
    fname = erase(infile, '.csv');
    % Copy to working dir with expected name
    copyfile(infile, strcat(ds, '.csv'));
    fname = ds;
    ftype = 2; % csv
    var_loc = [1,2,3,4]; % Time, Strain, Rate?, Stress (rate missing -> set to 0 below)
    var_loc(3) = 0; % differentiate strain to get rate
    var_conv = [1,1,1,1];
    data_trunc = [0,0,0];
    an_use = [1,1];
    omega = 2*pi; % matches generator
    M = 39; p = 1; k = 8; num_mode = 2;
    out_type = 1; is_fsf = 1; save_figs = 0;

    % Call SPPplus_fourier_v2 and numerical
    [time_wave,resp_wave,L,fname_t] = SPPplus_read_v2(fname,ftype,var_loc,var_conv,data_trunc);
    out_set=[out_type,is_fsf,save_figs];
    SPPplus_fourier_v2(time_wave,resp_wave,L,omega,M,p,out_set,fname_t);
    SPPplus_numerical_v2(time_wave,resp_wave,L,omega,k,num_mode,out_set,fname_t);

    % Move outputs into golden_data/outputs/matlab
    movefile(strcat(fname_t,'_SPP_FOURIER.txt'), fullfile(outdir, strcat(ds,'_spp_data_out_fourier.txt')));
    movefile(strcat(fname_t,'_SPP_FOURIER_FSFRAME.txt'), fullfile(outdir, strcat(ds,'_fsf_data_out_fourier.txt')));
    movefile(strcat(fname_t,'_SPP_FOURIER_PLOT.jpg'), fullfile(outdir, strcat(ds,'_plot_fourier.jpg')));
    movefile(strcat(fname_t,'_SPP_NUMERICAL.txt'), fullfile(outdir, strcat(ds,'_spp_data_out_numerical.txt')));
    movefile(strcat(fname_t,'_SPP_NUMERICAL_FSFRAME.txt'), fullfile(outdir, strcat(ds,'_fsf_data_out_numerical.txt')));
end

disp('MATLAB SPPplus runs complete. Outputs in golden_data/outputs/matlab');
