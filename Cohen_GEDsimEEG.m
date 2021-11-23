%%
% 
%  MATLAB code accompanying the paper:
%     A tutorial on generalized eigendecomposition for denoising, 
%     contrast enhancement, and dimension reduction in multichannel electrophysiology
% 
%   Mike X Cohen (mikexcohen@gmail.com)
%
% 
%  No MATLAB or 3rd party toolboxes are necessary.
%  The files emptyEEG.mat, filterFGx.m, and topoplotindie.m need to be in 
%    the MATLAB path or current directory.
% 
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%%

% load mat file containing EEG, leadfield and channel locations
load emptyEEG

% pick a dipole location in the brain
% It's fairly arbitrary; you can try different dipoles, although not 
% all dipoles have strong projections to the scalp electrodes.
diploc = 109;


% normalize dipoles (not necessary but simplifies the code)
lf.GainN = bsxfun(@times,squeeze(lf.Gain(:,1,:)),lf.GridOrient(:,1)') + bsxfun(@times,squeeze(lf.Gain(:,2,:)),lf.GridOrient(:,2)') + bsxfun(@times,squeeze(lf.Gain(:,3,:)),lf.GridOrient(:,3)');


% plot brain dipoles
figure(1), clf, subplot(221)
plot3(lf.GridLoc(:,1), lf.GridLoc(:,2), lf.GridLoc(:,3), 'o')
hold on
plot3(lf.GridLoc(diploc,1), lf.GridLoc(diploc,2), lf.GridLoc(diploc,3), 's','markerfacecolor','w','markersize',10)
rotate3d on, axis square, axis off
title('Brain dipole locations')


% Each dipole can be projected onto the scalp using the forward model. 
% The code below shows this projection from one dipole.
subplot(222)
topoplotIndie(lf.GainN(:,diploc), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
title('Signal dipole projection')


% Now we generate random data in brain dipoles.
% create 1000 time points of random data in brain dipoles
% (note: the '1' before randn controls the amount of noise)
dipole_data = 1*randn(length(lf.Gain),1000);

% add signal to second half of dataset
dipole_data(diploc,501:end) = 15*sin(2*pi*10*(0:499)/EEG.srate);

% project dipole data to scalp electrodes
EEG.data = lf.GainN*dipole_data;

% meaningless time series
EEG.times = (0:size(EEG.data,2)-1)/EEG.srate;

% plot the data from one channel
subplot(212), hold on
plot(EEG.times,dipole_data(diploc,:)/norm(dipole_data(diploc,:)),'linew',4)
plot(EEG.times,EEG.data(31,:)/norm(EEG.data(31,:)),'linew',2)
plot([.5 .5],get(gca,'ylim'),'k--','HandleVisibility','off');
xlabel('Time (s)'), ylabel('Amplitude (norm.)')
legend({'Dipole';'Electrode'})

%% Create covariance matrices

% compute covariance matrix R is first half of data
tmpd = EEG.data(:,1:500);
tmpd = bsxfun(@minus,tmpd,mean(tmpd,2));
covR = tmpd*tmpd'/500;

% compute covariance matrix S is second half of data
tmpd = EEG.data(:,501:end);
tmpd = bsxfun(@minus,tmpd,mean(tmpd,2));
covS = tmpd*tmpd'/500;


%%% plot the two covariance matrices
figure(2), clf

% S matrix
subplot(131)
imagesc(covS)
title('S matrix')
axis square, set(gca,'clim',[-1 1]*1e6)

% R matrix
subplot(132)
imagesc(covR)
title('R matrix')
axis square, set(gca,'clim',[-1 1]*1e6)

% R^{-1}S
% Note: GED doesn't require the explicit inverse of R;
%       it's here only for visualization
subplot(133)
imagesc(inv(covR)*covS)
title('R^-^1S matrix')
axis square, set(gca,'clim',[-10 10])


%% ----------------------------- %%
%                                 %
%  Dimension compression via PCA  %
%                                 %
%%% --------------------------- %%%

% This code cell demonstrates that PCA is unable
% recover the simulated dipole signal.

% PCA
[evecs,evals] = eig( (covS+covR)/2 );

% sort eigenvalues/vectors
[evals,sidx] = sort(diag(evals),'descend');
evecs = evecs(:,sidx);



% plot the eigenspectrum
figure(3), clf
subplot(231)
plot(evals,'ks-','markersize',10,'markerfacecolor','r')
axis square
set(gca,'xlim',[0 20.5])
title('PCA eigenvalues')
xlabel('Component number'), ylabel('Power ratio (\lambda)')


% component time series is eigenvector as spatial filter for data
comp_ts = evecs(:,1)'*EEG.data;


% normalize time series (for visualization)
dipl_ts = dipole_data(diploc,:) / norm(dipole_data(diploc,:));
comp_ts = comp_ts / norm(comp_ts);
chan_ts = EEG.data(31,:) / norm(EEG.data(31,:));


% plot the time series
subplot(212), hold on
plot(EEG.times,.3+dipl_ts,'linew',2)
plot(EEG.times,.15+chan_ts)
plot(EEG.times,comp_ts)
legend({'Truth';'EEG channel';'PCA time series'})
set(gca,'ytick',[])
xlabel('Time (a.u.)')


%% spatial filter forward model

% The filter forward model is what the source "sees" when it looks through the
% electrodes. It is obtained by passing the covariance matrix through the filter.
filt_topo = evecs(:,1);

% Eigenvector sign uncertainty can cause a sign-flip, which is corrected for by 
% forcing the largest-magnitude projection electrode to be positive.
[~,se] = max(abs( filt_topo ));
filt_topo = filt_topo * sign(filt_topo(se));


% plot the maps
subplot(232)
topoplotIndie(lf.GainN(:,diploc), EEG.chanlocs,'numcontour',0,'electrodes','off','shading','interp');
title('Truth topomap')

subplot(233)
topoplotIndie(filt_topo,EEG.chanlocs,'electrodes','off','numcontour',0);
title('PCA forward model')

%% ----------------------------- %%
%                                 %
%    Source separation via GED    %
%                                 %
%%% --------------------------- %%%


% Generalized eigendecomposition (GED)
[evecs,evals] = eig(covS,covR);

% sort eigenvalues/vectors
[evals,sidx] = sort(diag(evals),'descend');
evecs = evecs(:,sidx);



% plot the eigenspectrum
figure(4), clf
subplot(231)
plot(evals,'ks-','markersize',10,'markerfacecolor','m')
axis square
set(gca,'xlim',[0 20.5])
title('GED eigenvalues')
xlabel('Component number'), ylabel('Power ratio (\lambda)')

% component time series is eigenvector as spatial filter for data
comp_ts = evecs(:,1)'*EEG.data;

%% plot for comparison

% normalize time series (for visualization)
dipl_ts = dipole_data(diploc,:) / norm(dipole_data(diploc,:));
comp_ts = comp_ts / norm(comp_ts);
chan_ts = EEG.data(31,:) / norm(EEG.data(31,:));


% plot the time series
subplot(212), hold on
plot(EEG.times,.3+dipl_ts,'linew',2)
plot(EEG.times,.15+chan_ts)
plot(EEG.times,comp_ts)
legend({'Truth';'EEG channel';'GED time series'})
set(gca,'ytick',[])
xlabel('Time (a.u.)')


%% spatial filter forward model

% The filter forward model is what the source "sees" when it looks through the
% electrodes. It is obtained by passing the covariance matrix through the filter.
filt_topo = covS*evecs(:,1);


% Eigenvector sign uncertainty can cause a sign-flip, which is corrected for by 
% forcing the largest-magnitude projection electrode to be positive.
[~,se] = max(abs( filt_topo ));
filt_topo = filt_topo * sign(filt_topo(se));


% plot the maps
subplot(232)
topoplotIndie(lf.GainN(:,diploc), EEG.chanlocs,'numcontour',0,'electrodes','off','shading','interp');
title('Truth topomap')

subplot(233)
topoplotIndie(filt_topo,EEG.chanlocs,'electrodes','off','numcontour',0);
title('GED forward model')


%% ICA

% NOTE: This cell computes ICA based on the jade algorithm. It's not
% discussed or shown in the paper, but you can uncomment this section if
% you are curious. Make sure the jader() function is in the MATLAB path
% (you can download it from the web if you don't have it).

% ivecs = jader(EEG.data,40);
% ic_scores = ivecs*EEG.data;
% icmaps = pinv(ivecs');
% evals = diag(icmaps*icmaps');
% 
% 
% % plot the IC energy
% figure(5), clf
% subplot(231)
% plot(evals,'ks-','markersize',10,'markerfacecolor','m')
% axis square
% set(gca,'xlim',[0 20.5])
% title('ICA RMS')
% xlabel('Component number'), ylabel('IC energy')
% 
% % component time series is eigenvector as spatial filter for data
% comp_ts = ic_scores(1,:);%evecs(:,1)'*EEG.data;
% 
% % plot for comparison
% 
% % normalize time series (for visualization)
% dipl_ts = dipole_data(diploc,:) / norm(dipole_data(diploc,:));
% comp_ts = comp_ts / norm(comp_ts);
% chan_ts = EEG.data(31,:) / norm(EEG.data(31,:));
% 
% 
% % plot the time series
% subplot(212), hold on
% plot(EEG.times,.3+dipl_ts,'linew',2)
% plot(EEG.times,.15+chan_ts)
% plot(EEG.times,comp_ts)
% legend({'Truth';'EEG channel';'ICA time series'})
% set(gca,'ytick',[])
% xlabel('Time (a.u.)')
% 
% 
% % plot the maps
% subplot(232)
% topoplotIndie(lf.GainN(:,diploc), EEG.chanlocs,'numcontour',0,'electrodes','off','shading','interp');
% title('Truth topomap')
% 
% subplot(233)
% topoplotIndie(icmaps(1,:),EEG.chanlocs,'electrodes','off','numcontour',0);
% title('ICA forward model')


%% ----------------------------- %%
%                                 %
%   Example GED in richer data    %
%                                 %
%%% --------------------------- %%%

% The above simulation is overly simplistic. The goal of
% this section is to simulate data that shares more 
% characteristics to real EEG data, including non-sinusoidal
% rhythms, background noise, and multiple trials.

% This code will simulate resting-state that has been segmented
% into 2-second non-overlapping epochs.


% signal parameters in Hz
peakfreq = 10; % "alpha"
fwhm     =  5; % full-width at half-maximum around the alpha peak


% EEG parameters for the simulation
EEG.srate  = 500; % sampling rate in Hz
EEG.pnts   = 2*EEG.srate; % each data segment is 2 seconds
EEG.trials = 50;


%%% create frequency-domain Gaussian
hz = linspace(0,EEG.srate,EEG.pnts);
s  = fwhm*(2*pi-1)/(4*pi); % normalized width
x  = hz-peakfreq;          % shifted frequencies
fg = exp(-.5*(x/s).^2);    % gaussian



% loop over trials and generate data
for triali=1:EEG.trials
    
    % random Fourier coefficients
    fc = rand(1,EEG.pnts) .* exp(1i*2*pi*rand(1,EEG.pnts));
    
    % taper with the Gaussian
    fc = fc .* fg;
    
    % back to time domain to get the source activity
    source_ts = 2*real( ifft(fc) )*EEG.pnts;
    dipole_ts(:,triali) = source_ts;
    
    % simulate dipole data: all noise and replace target dipole with source_ts
    dipole_data = randn(length(lf.GainN),EEG.pnts);
    dipole_data(diploc,:) = .5*source_ts;
    % Note: the source time series has low amplitude to highlight the
    % sensitivity of GED. Increasing this gain to, e.g., 1 will show
    % accurate though noiser reconstruction in the channel data.
    
    % now project the dipole data through the forward model to the electrodes
    EEG.data(:,:,triali) = lf.GainN*dipole_data;
end


% power spectrum of the ground-truth source activity
sourcepowerAve = mean(abs(fft(dipole_ts,[],1)).^2,2);

%% topoplot of alpha power

channelpower = abs(fft(EEG.data,[],2)).^2;
channelpowerAve = squeeze(mean(channelpower,3));

% vector of frequencies
hz = linspace(0,EEG.srate/2,floor(EEG.pnts/2)+1);

%% Create a covariance tensor (one covmat per trial)

% filter the data around 10 Hz
alphafilt = filterFGx(EEG.data,EEG.srate,10,4);

% initialize covariance matrices (one for each trial)
[allCovS,allCovR] = deal( zeros(EEG.trials,EEG.nbchan,EEG.nbchan) );

% loop over trials (data segments) and compute each covariance matrix
for triali=1:EEG.trials
    
    % cut out a segment
    tmpdat = alphafilt(:,:,triali);
    
    % mean-center
    tmpdat = tmpdat-mean(tmpdat,2);
    
    % add to S tensor
    allCovS(triali,:,:) = tmpdat*tmpdat' / EEG.pnts;
    
    % repeat for broadband data
    tmpdat = EEG.data(:,:,triali);
    tmpdat = tmpdat-mean(tmpdat,2);
    allCovR(triali,:,:) = tmpdat*tmpdat' / EEG.pnts;
end
    
%%% illustration of cleaning covariance matrices

% clean R
meanR = squeeze(mean(allCovR));  % average covariance
dists = zeros(EEG.trials,1);     % vector of distances to mean
for segi=1:size(allCovR,1)
    r = allCovR(segi,:,:);
    % Euclidean distance
    dists(segi) = sqrt( sum((r(:)-meanR(:)).^2) );
end

% finally, average trial-covariances together, excluding outliers
covR = squeeze(mean( allCovR(zscore(dists)<3,:,:) ,1));


%%%%% Normally you'd repeat the above for S; ommitted here for simplicity
covS = squeeze(mean( allCovS ,1));

%% now for the GED

%%% NOTE: You can test PCA on these data by using only covS, or only covR,
%         in the eig() function.

% eig and sort
[evecs,evals] = eig(covS,covR);
[evals,sidx]  = sort(diag(evals),'descend');
evecs = evecs(:,sidx);

%%% compute the component time series
% for the multiplication, the data need to be reshaped into 2D
data2D = reshape(EEG.data,EEG.nbchan,[]);
compts = evecs(:,1)' * data2D;
% and then reshaped back into trials
compts = reshape(compts,EEG.pnts,EEG.trials);

%%% power spectrum
comppower = abs(fft(compts,[],1)).^2;
comppowerAve = squeeze(mean(comppower,2));

%%% component map
compmap = evecs(:,1)' * covS;
% flip map sign
[~,se] = max(abs( compmap ));
compmap = compmap * sign(compmap(se));

%% visualization


figure(5), clf

subplot(241)
topoplotIndie(lf.GainN(:,diploc), EEG.chanlocs,'numcontour',0,'electrodes','off','shading','interp');
title('Truth topomap')
set(gca,'clim',[-1 1]*30)

subplot(242)
plot(evals,'ks-','markersize',10,'markerfacecolor','r')
axis square, box off
set(gca,'xlim',[0 20.5])
title('GED scree plot')
xlabel('Component number'), ylabel('Power ratio (\lambda)')
% Note that the max eigenvalue is <1, 
% because R has more overall energy than S.

subplot(243)
topoplotIndie(compmap,EEG.chanlocs,'numcontour',0);
title('Alpha component')
set(gca,'clim',[-1 1]*10)

subplot(244)
topoplotIndie(channelpowerAve(:,dsearchn(hz',10)),EEG.chanlocs,'numcontour',0);%,'electrodes','numbers');
title('Elecr. power (10 Hz)')
set(gca,'clim',[1e8 1.5e9])




subplot(212), cla, hold on
plot(hz,sourcepowerAve(1:length(hz))/max(sourcepowerAve(1:length(hz))),'m','linew',3)
plot(hz,comppowerAve(1:length(hz))/max(comppowerAve(1:length(hz))),'r','linew',3)
plot(hz,channelpowerAve(31,1:length(hz))/max(channelpowerAve(31,1:length(hz))),'k','linew',3)
legend({'Source','Component','Electrode 31'},'box','off')
xlabel('Frequency (Hz)')
ylabel('Power (norm to max power)')
set(gca,'xlim',[0 80])

% font size for all axes
set(get(gcf,'children'),'fontsize',13)

%% done.
