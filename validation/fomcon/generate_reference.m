%% generate_reference.m
% Reference data from FOMCON (MATLAB) for the cross-validation of pyfopid.
%
% Requirements: MATLAB, Control System Toolbox and FOMCON
% (https://fomcon.net, https://github.com/extall/fomcon-matlab).
%
% Usage (from this folder):
%   >> generate_reference
%
% Output (in ./data):
%   environment.txt            MATLAB and toolbox versions
%   case1_oustaloup.csv        Oustaloup approximation of s^alpha (FOMCON oustapp)
%   case2_fopid_params.csv     FOPID parameters and sampling step
%   case2_fopid_response.csv   FOPID controller response u(t) to a test error e(t)
%   log.txt                    Which cases ran and any error messages
%
% The Python test tests/test_fomcon_reference.py reads these files.

clear; clc;
here   = fileparts(mfilename('fullpath'));
outdir = fullfile(here, 'data');
if ~exist(outdir, 'dir'); mkdir(outdir); end

logf = fopen(fullfile(outdir, 'log.txt'), 'w');
logmsg = @(varargin) fprintf(logf, [varargin{1} '\n'], varargin{2:end});

%% Environment ------------------------------------------------------------
fid = fopen(fullfile(outdir, 'environment.txt'), 'w');
fprintf(fid, 'MATLAB %s\n', version);
v = ver;
for k = 1:numel(v)
    fprintf(fid, '%s %s\n', v(k).Name, v(k).Version);
end
fprintf(fid, 'fotf: %s\n', which('fotf'));
fprintf(fid, 'oustapp: %s\n', which('oustapp'));
fprintf(fid, 'fracpid: %s\n', which('fracpid'));
fclose(fid);

%% Case 1: Oustaloup approximation of s^alpha ------------------------------
% Same band and order for every alpha; frequency response on a log grid.
try
    alphas = [-0.9 -0.5 -0.3 0.3 0.5 0.8];
    wb = 1e-3; wh = 1e3; N = 5;
    w  = logspace(-3, 3, 121);
    rows = [];
    for a = alphas
        if a >= 0
            G = fotf(1, 0, 1, a);    % s^a      (fotf(a, na, b, nb) = b s^nb / a s^na)
        else
            G = fotf(1, -a, 1, 0);   % 1 / s^-a
        end
        Z = oustapp(G, wb, wh, N, 'oust');
        H = squeeze(freqresp(Z, w));
        rows = [rows; repmat([a wb wh N], numel(w), 1), w(:), ...
                20*log10(abs(H(:))), angle(H(:))*180/pi]; %#ok<AGROW>
    end
    write_csv(fullfile(outdir, 'case1_oustaloup.csv'), ...
        'alpha,wb,wh,N,w,mag_db,phase_deg', rows);
    logmsg('case1 OK (%d rows)', size(rows, 1));
catch err
    logmsg('case1 FAILED: %s', err.message);
end

%% Case 2: FOPID controller response to a known error signal ---------------
% u = C(s) e,  C(s) = Kp + Ki s^-lambda + Kd s^mu, simulated with FOMCON lsim.
try
    Kp = 1.2; Ki = 0.8; lam = 0.7; Kd = 0.3; mu = 0.6;
    dt = 1e-3; T = 5;
    t  = (0:dt:T)';
    e  = sin(2*t) + double(t >= 1);
    C  = fracpid(Kp, Ki, lam, Kd, mu);
    u  = lsim(C, e, t);
    write_csv(fullfile(outdir, 'case2_fopid_params.csv'), ...
        'kp,ki,lam,kd,mu,dt', [Kp Ki lam Kd mu dt]);
    write_csv(fullfile(outdir, 'case2_fopid_response.csv'), ...
        't,e,u', [t e u(:)]);
    logmsg('case2 OK (%d samples)', numel(t));
catch err
    logmsg('case2 FAILED: %s', err.message);
end

fclose(logf);
type(fullfile(outdir, 'log.txt'));

%% ------------------------------------------------------------------------
function write_csv(path, header, M)
    fid = fopen(path, 'w');
    fprintf(fid, '%s\n', header);
    fmt = [repmat('%.12g,', 1, size(M, 2) - 1) '%.12g\n'];
    fprintf(fid, fmt, M.');
    fclose(fid);
end
