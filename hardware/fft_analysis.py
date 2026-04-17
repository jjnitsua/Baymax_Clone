import numpy as np

SAMPLE_RATE_HZ = 100

def compute_fft(x_samples, y_samples, z_samples):
    """
    Compute FFT on a batch of samples and return
    frequency amplitudes within a target band.

    Args:
        x_samples, y_samples, z_samples: lists of floats (acceleration in G)

    Returns:
        dict with frequencies and amplitudes per axis, plus band summary
    """
    n = len(x_samples)                        # number of samples in batch

    # --- convert to numpy arrays ---
    x_arr = np.array(x_samples)
    y_arr = np.array(y_samples)
    z_arr = np.array(z_samples)

    # --- compute frequency axis ---
    # rfftfreq returns the frequencies corresponding to each FFT output bin
    # e.g. at 100 Hz with 25 samples: bins at 0, 4, 8, 12 ... 50 Hz
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE_HZ)

    # --- compute FFT per axis ---
    # rfft = real FFT, used because our input is real-valued (not complex)
    # np.abs() converts complex FFT output to amplitude
    # * 2 / n normalises amplitude to actual G units
    x_amp = np.abs(np.fft.rfft(x_arr)) * 2 / n
    y_amp = np.abs(np.fft.rfft(y_arr)) * 2 / n
    z_amp = np.abs(np.fft.rfft(z_arr)) * 2 / n

    return freqs, x_amp, y_amp, z_amp


def band_summary(freqs, x_amp, y_amp, z_amp, low_hz=4.0, high_hz=7.0):
    """
    Extract amplitude information within a specific frequency band.

    Args:
        freqs:            frequency axis from compute_fft()
        x_amp, y_amp, z_amp: amplitude arrays from compute_fft()
        low_hz, high_hz:  band of interest in Hz

    Returns:
        dict with peak frequency and amplitude per axis within the band
    """
    # boolean mask — True for bins that fall inside the band
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)

    # frequencies and amplitudes inside the band only
    band_freqs  = freqs[band_mask]
    band_x      = x_amp[band_mask]
    band_y      = y_amp[band_mask]
    band_z      = z_amp[band_mask]

    def axis_summary(band_f, band_a):
        if len(band_a) == 0:
            return {"peak_hz": None, "peak_amp_g": None, "mean_amp_g": None}
        peak_idx = np.argmax(band_a)           # index of strongest frequency
        return {
            "peak_hz":    round(float(band_f[peak_idx]), 2),   # frequency of peak
            "peak_amp_g": round(float(band_a[peak_idx]), 6),   # amplitude at peak
            "mean_amp_g": round(float(np.mean(band_a)),  6),   # mean across band
        }

    return {
        "band_hz": [low_hz, high_hz],
        "x": axis_summary(band_freqs, band_x),
        "y": axis_summary(band_freqs, band_y),
        "z": axis_summary(band_freqs, band_z),
    }