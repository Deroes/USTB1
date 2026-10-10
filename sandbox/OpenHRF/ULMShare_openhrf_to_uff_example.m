%% Read OpenH-RF ULMShare plane-wave IQ data into USTB
%
% Converts one ULMShare acquisition (transcranial mouse-brain ULM, Verasonics
% Vantage 256 with an L22-14v probe, plane-wave IQ) from the OpenH-RF zea
% format to a uff.channel_data object, beamforms it with USTB DAS on the
% ULMShare reference grid, and shows a B-mode image and a power-Doppler image
% after SVD clutter filtering.
%
% Dataset: https://huggingface.co/datasets/nvidia/OpenH-RF/tree/main/ulmshare
% Explorer: https://open-h.github.io/OpenH-RF/#dataset/ulmshare
% Citation: Rauby et al. (2026), ULMShare, doi:10.20383/103.01550 (CC BY 4.0)
%
% The full acquisitions are 60-370 GB, so first stream one 400-frame buffer
% (~720 MB local) with the helper in this folder:
%
%   python openhrf_download_subset.py ^
%     --source hf://nvidia/OpenH-RF/ulmshare/data/mouse_10_acquisition_1_zea.hdf5 ^
%     --output C:/Data/OpenH-RF/ulmshare/mouse_10_acquisition_1_frames_20000_20399.hdf5 ^
%     --first-frame 20000 --n-frames 400
%
% (needs: pip install h5py hdf5plugin huggingface_hub numpy)
%
% Frame 20000 (buffer 50) is used because the first buffers of an acquisition
% are typically recorded before the microbubble bolus arrives (e.g. frames
% 8000-8399 of mouse 10 show tissue and noise only in power Doppler).

clear all; close all;

local_file = 'C:/Data/OpenH-RF/ulmshare/mouse_10_acquisition_1_frames_20000_20399.hdf5';

%% Convert to UFF
channel_data = openhrf_to_uff(local_file);
disp(channel_data.name.');

%% ULMShare reference beamforming grid (25 um pixels)
scan = uff.linear_scan();
scan.x_axis = (-5e-3:25e-6:5e-3-25e-6).';
scan.z_axis = (0.5e-3:25e-6:8.5e-3-25e-6).';

%% DAS, compounding all plane waves
das = midprocess.das();
das.channel_data = channel_data;
das.scan = scan;
das.dimension = dimension.both;
das.receive_apodization.window = uff.window.hanning;
das.receive_apodization.f_number = 1.4;
das.transmit_apodization.window = uff.window.none;
b_data = das.go();

%% B-mode of the first frame
b_mode = uff.beamformed_data(b_data);
b_mode.data = b_data.data(:, :, :, 1);
b_mode.plot([], 'ULMShare mouse 10 (OpenH-RF) - B-mode', 50);

%% Power Doppler after SVD clutter filtering of the beamformed frames
n_frames = b_data.N_frames;
casorati = reshape(b_data.data, [], n_frames);              % [pixel x frame]
n_cut = max(round(0.05*n_frames) - 1, 0);                    % ULMShare: reject 5% of the components
[U, S, V] = svd(casorati, 'econ');
filtered = casorati - U(:, 1:n_cut)*S(1:n_cut, 1:n_cut)*V(:, 1:n_cut)';
power_doppler = reshape(mean(abs(filtered).^2, 2), scan.N_z_axis, scan.N_x_axis);

figure;
imagesc(scan.x_axis*1e3, scan.z_axis*1e3, 10*log10(power_doppler./max(power_doppler(:))));
axis image; colormap hot; colorbar; caxis([-40 0]);
xlabel('x [mm]'); ylabel('z [mm]');
title(sprintf('ULMShare mouse 10 (OpenH-RF) - power Doppler, %d frames', n_frames));
