lyaps_fourier.py \
--in-dir ./p1d_out/p1d_lya_noisepipeline_linesDESIEDR_catiron_highz_20240305_dlairon_highz_CNN_GP_naim
cuts_balai/Delta \
--in-format fits \
--out-dir ./p1d_out/p1d_lya_noisepipeline_linesDESIEDR_catiron_highz_20240305_dlairon_highz_CNN_GP_nai
mcuts_balai/pk1d_SNRcut1 \
--out-format fits \
--lambda-obs-min 3750. \
--noise-estimate pipeline \
--num-processors 128 \
--nb-pixel-masked-max 120 \
--nb-part 3 \
--SNR-min 1 \
--num-noise-exp 2500


lyapsfft1d_average.py \
--in-dir ./p1d_out/p1d_lya_noisepipeline_linesDESIEDR_catiron_highz_20240305_dlairon_highz_CNN_GP_naim
cuts_balai/pk1d_SNRcut1 \
--weight-method fit_snr \
--ncpu 128 \
--velunits \
--rebinfac 1  \
--covariance \
--bootstrap \
--nbootstrap 50 \
--bootstrap_average \
--overwrite