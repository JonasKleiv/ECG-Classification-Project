import gc
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import neurokit2 as nk
import numpy as np
import pandas as pd
import pywt
from tqdm import tqdm

# ==============================================================================
# 1. Clinical Constants & Feature Names (Default Configuration)
# ==============================================================================
DEFAULT_CLINICAL_CONFIG = {
    "ref_lead_idx": 1,
    "st_offset_ms": 60,
    "qrs_range_ms": (40, 200),
    "qt_range_ms":  (200, 600),
    "pr_range_ms":  (80, 400),
    "p_range_ms":   (40, 200),
    "rr_range_sec": (0.3, 2.0),
}

# Keys for ECG lead features and rhythm features
LEAD_FEATURE_KEYS = [
    "qrs_duration",
    "qt_duration",
    "p_duration",
    "pr_duration",
    "r_amplitude",
    "t_amplitude",
    "st_shift",
]

# Keys for ECG rhythm features
RYTHM_FEATURE_KEYS = [
    "heart_rate",
    "heart_rate_std",
]

# ==============================================================================
# 2. PQRST Clinical Feature Extraction Helpers
# ==============================================================================
def is_missing(value):
    return np.isnan(value)

def median_of(values):
    if len(values) == 0:
        return np.nan
    return float(np.median(values))

def is_plausible(value, minimum, maximum):
    return minimum <= value <= maximum

# Extract landmarks per beat for a given key
def get_landmarks_per_beat(waves, key, n_beats):
    raw_values = waves.get(key)

    if raw_values is None or len(raw_values) != n_beats:
        return np.full((n_beats,), np.nan)

    values = []
    for item in raw_values:
        if item is None:
            values.append(np.nan)
        else:
            values.append(item)

    return values

# Extract rhythm features from R-peaks and sampling rate
def extract_rythm_features(r_peaks, sampling_rate, cfg):
    features = {}

    features["heart_rate"] = np.nan
    features["heart_rate_std"] = np.nan

    # Calculate heart rate for each R-R interval within plausible range
    heart_rates = []
    for i in range(len(r_peaks) - 1):
        samples_between_beats = r_peaks[i + 1] - r_peaks[i]
        r_r_seconds = samples_between_beats / sampling_rate

        if not is_plausible(r_r_seconds, cfg["rr_range_sec"][0], cfg["rr_range_sec"][1]):
            continue

        beats_per_minute = 60 / r_r_seconds
        heart_rates.append(beats_per_minute)

    if len(heart_rates) >= 1:
        features["heart_rate"] = np.mean(heart_rates)
    if len(heart_rates) >= 2:
        features["heart_rate_std"] = np.std(heart_rates)

    return features

# Extract clinical features from a single ECG lead
def extract_lead_clinical_features(signal, r_peaks, sampling_rate, cfg=DEFAULT_CLINICAL_CONFIG, verbose=0):
    n_samples = len(signal)
    n_beats = len(r_peaks)

    ms_per_sample = 1000 / sampling_rate

    # Delineate ECG waves to extract landmarks per beat
    _, waves = nk.ecg_delineate(
        signal, 
        rpeaks=r_peaks, 
        sampling_rate=sampling_rate, 
        method="dwt"
    )

    if verbose > 0:
        print("Delineation completed. Extracted waves:", list(waves.keys()))
        print("Number of beats:", n_beats)
        print("Example wave landmarks for first beat:", {key: waves[key][0] for key in waves})

    # Extract landmarks for each wave type per beat
    p_onsets = get_landmarks_per_beat(waves, "ECG_P_Onsets", n_beats)
    p_offsets = get_landmarks_per_beat(waves, "ECG_P_Offsets", n_beats)
    qrs_onsets = get_landmarks_per_beat(waves, "ECG_R_Onsets", n_beats)
    qrs_offsets = get_landmarks_per_beat(waves, "ECG_R_Offsets", n_beats)
    t_peaks = get_landmarks_per_beat(waves, "ECG_T_Peaks", n_beats)
    t_offsets = get_landmarks_per_beat(waves, "ECG_T_Offsets", n_beats)

    if verbose > 1:
        print("Extracted landmarks per beat:")
        print("P onsets:", p_onsets)
        print("P offsets:", p_offsets)
        print("QRS onsets:", qrs_onsets)
        print("QRS offsets:", qrs_offsets)
        print("T peaks:", t_peaks)
        print("T offsets:", t_offsets)

    qrs_durations = []
    qt_durations = []
    p_durations = []
    pr_durations = []
    r_amplitudes = []
    t_amplitudes = []
    st_shifts = []

    st_offset_in_samples = int(round(cfg["st_offset_ms"] / ms_per_sample))

    # Iterate over each beat to calculate clinical features
    for beat in range(n_beats):
        p_onset = p_onsets[beat]
        p_offset = p_offsets[beat]
        qrs_onset = qrs_onsets[beat]
        qrs_offset = qrs_offsets[beat]
        t_peak = t_peaks[beat]
        t_offset = t_offsets[beat]

        # QRS duration
        if not is_missing(qrs_onset) and not is_missing(qrs_offset):
            duration_ms = (qrs_offset - qrs_onset) * ms_per_sample
            if is_plausible(duration_ms, cfg["qrs_range_ms"][0], cfg["qrs_range_ms"][1]):
                qrs_durations.append(duration_ms)

        # QT duration
        if not is_missing(qrs_onset) and not is_missing(t_offset):
            duration_ms = (t_offset - qrs_onset) * ms_per_sample
            if is_plausible(duration_ms, cfg["qt_range_ms"][0], cfg["qt_range_ms"][1]):
                qt_durations.append(duration_ms)

        # P duration
        if not is_missing(p_onset) and not is_missing(p_offset):
            duration_ms = (p_offset - p_onset) * ms_per_sample
            if is_plausible(duration_ms, cfg["p_range_ms"][0], cfg["p_range_ms"][1]):
                p_durations.append(duration_ms)

        # PR duration
        if not is_missing(p_onset) and not is_missing(qrs_onset):
            duration_ms = (qrs_onset - p_onset) * ms_per_sample
            if is_plausible(duration_ms, cfg["pr_range_ms"][0], cfg["pr_range_ms"][1]):
                pr_durations.append(duration_ms)

        # R amplitude
        r_sample = int(r_peaks[beat])
        if 0 <= r_sample <= n_samples:
            r_amplitudes.append(signal[r_sample])

        # T amplitude
        if not is_missing(t_peak):
            t_sample = int(t_peak)
            if 0 <= t_sample <= n_samples:
                t_amplitudes.append(signal[t_sample])

        # ST shift
        have_landmarks = (
            not is_missing(qrs_offset)
            and not is_missing(p_offset)
            and not is_missing(qrs_onset)
        )

        # Check if all necessary landmarks are present for ST shift calculation
        if have_landmarks:
            j_point = int(qrs_offset)
            baseline_start = int(p_offset)
            baseline_end = int(qrs_onset)
            st_sample = j_point + st_offset_in_samples

            baseline_is_valid = 0 <= baseline_start <= baseline_end
            st_sample_is_inside_signal = st_sample < n_samples
            
            if baseline_is_valid and st_sample_is_inside_signal:
                baseline_level = np.mean(signal[baseline_start:baseline_end])
                st_level = signal[st_sample]
                st_shifts.append(st_level - baseline_level)

    features = {}
    features["qrs_duration"] = median_of(qrs_durations)
    features["qt_duration"] = median_of(qt_durations)
    features["p_duration"] = median_of(p_durations)
    features["pr_duration"] = median_of(pr_durations)
    features["r_amplitude"] = median_of(r_amplitudes)
    features["t_amplitude"] = median_of(t_amplitudes)
    features["st_shift"] = median_of(st_shifts)

    return features

# Find R peaks in the ECG signals using the reference lead first, then fallback to other leads if necessary
def find_r_peaks(cleaned_leads, sampling_rate, cfg):
    lead_order = [cfg["ref_lead_idx"]]

    for lead in range(len(cleaned_leads)):
        if lead != cfg["ref_lead_idx"]:
            lead_order.append(lead)

    for lead in lead_order:
        signal = cleaned_leads[lead]
        if signal is None:
            continue

        try:
            _, peak_info = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
        except Exception:
            continue

        found_peaks = peak_info.get("ECG_R_Peaks", [])

        valid_peaks = []
        for peak in found_peaks:
            if np.isnan(peak):
                continue
            peak = int(peak)
            if 0 <= peak < len(signal):
                valid_peaks.append(peak)
        
        if len(valid_peaks) >= 2:
            return np.array(valid_peaks)

    return None

_worker_mmap = None
_worker_file = None

# Process a single patient's ECG signals from a memory-mapped file and extract features
def _process_patient_from_file(patient_idx, npy_file, sampling_rate, cfg):
    global _worker_mmap, _worker_file
    if _worker_mmap is None or _worker_file != npy_file:
        _worker_mmap = np.load(npy_file, mmap_mode="r")
        _worker_file = npy_file

    # Access the patient's signals from the memory-mapped file
    patient_signals = np.asarray(_worker_mmap[patient_idx], dtype=np.float64)
    n_leads = patient_signals.shape[1]
    n_features_total = len(RYTHM_FEATURE_KEYS) + n_leads * len(LEAD_FEATURE_KEYS)
    n_failures = 0

    # Clean the ECG leads before feature extraction
    cleaned_leads = [patient_signals[:, lead] for lead in range(n_leads)]

    # Find R peaks in the cleaned ECG leads
    r_peaks = find_r_peaks(cleaned_leads, sampling_rate, cfg)
    if r_peaks is None:
        return [np.nan] * n_features_total, n_leads, True

    # Extract rhythm features based on the detected R peaks
    row = []
    rythm_features = extract_rythm_features(r_peaks, sampling_rate, cfg)
    for key in RYTHM_FEATURE_KEYS:
        row.append(rythm_features[key])

    # Extract lead-specific features for each ECG lead
    for lead in range(n_leads):
        try:
            lead_features = extract_lead_clinical_features(
                cleaned_leads[lead], r_peaks, sampling_rate, cfg=cfg
            )
        except Exception:
            n_failures += 1
            lead_features = {k: np.nan for k in LEAD_FEATURE_KEYS}
        for key in LEAD_FEATURE_KEYS:
            row.append(lead_features[key])
    return row, n_failures, False

def extract_pqrst_dataset(
    filtered_npy_path,
    output_csv_path, 
    sampling_rate, 
    clinical_config=None,
    force_reload=False
):
    """
    Extract 86 clinical PQRST + rhythm features across all 12 leads in parallel,
    or load directly from CSV cache if available.
    """
    cfg = {**DEFAULT_CLINICAL_CONFIG, **(clinical_config or {})}

    # Check if the output CSV already exists and load it if not forcing reload
    if os.path.exists(output_csv_path) and not force_reload:
        print(f"Loading cached PQRST features from: {output_csv_path}")
        return pd.read_csv(output_csv_path)

    # Load the filtered ECG signals from the memory-mapped NumPy file
    X = np.load(filtered_npy_path, mmap_mode="r")
    num_patients, _, num_leads = X.shape
    n_workers = os.cpu_count()
    print(f"Extracting PQRST features for {num_patients} patients using {n_workers} workers...")
    
    # Process each patient in parallel to extract PQRST features
    results = Parallel(n_jobs=n_workers, return_as="generator", batch_size=8)(
        delayed(_process_patient_from_file)(idx, filtered_npy_path, sampling_rate, cfg)
        for idx in range(num_patients)
    )
    
    all_rows = []
    total_failures = 0
    patients_without_peaks = 0

    # Aggregate the results from parallel processing
    for row, n_failures, no_peaks in tqdm(
        results, total=num_patients, desc="Extracting PQRST features"
    ):
        all_rows.append(row)
        total_failures += n_failures
        if no_peaks:
            patients_without_peaks += 1

    # Define the column names for the resulting DataFrame
    column_names = list(RYTHM_FEATURE_KEYS) + [
        f"lead{lead_idx + 1}_{key}"
        for lead_idx in range(num_leads)
        for key in LEAD_FEATURE_KEYS
    ]

    df_pqrst = pd.DataFrame(all_rows, columns=column_names)
    df_pqrst.to_csv(output_csv_path, index=False)

    # Print summary of the extraction process
    print(f"Saved PQRST features to {output_csv_path} (shape={df_pqrst.shape})")
    print(f"Patients with no usable R-peaks: {patients_without_peaks}/{num_patients}")
    print(f"Lead-level delineation failures: {total_failures}/{num_patients * num_leads}")
    
    return df_pqrst

# ==============================================================================
# 3. Discrete Wavelet Transform (DWT) Feature Extraction
# ==============================================================================
# Helper function to calculate Shannon entropy of a coefficient array
def calc_shannon_entropy(coefficient):
    energy = np.square(coefficient)
    energy_sum = np.sum(energy)
    if energy_sum == 0:
        return 0.0
    probabilities = energy / energy_sum
    return float(-np.sum(probabilities * np.log2(probabilities + 1e-12)))

# Helper function to extract DWT features from a single lead signal
def extract_single_lead_dwt(signal, wavelet="db4", level=3):
    coefficients = pywt.wavedec(signal, wavelet=wavelet, level=level)
    features = []
    for coeff in coefficients:
        features.extend([
            np.var(coeff),
            np.sum(np.square(coeff)),
            calc_shannon_entropy(coeff),
            np.max(coeff),
            np.min(coeff),
        ])
    return features

# Main function to extract DWT features for the entire dataset
def extract_dwt_dataset(
    filtered_npy_path,
    output_csv_path,
    wavelet="db4",
    level=3,
    force_reload=False,
):
    """
    Extract 240 statistical DWT features across all 12 leads,
    or load directly from CSV cache if available.
    """
    # Check if cached CSV exists and load it if available
    if os.path.exists(output_csv_path) and not force_reload:
        print(f"Loading cached DWT features from: {output_csv_path}")
        return pd.read_csv(output_csv_path)

    # Load the filtered ECG signals from the .npy file
    X_clean = np.load(filtered_npy_path, mmap_mode="r")
    num_patients, _, num_leads = X_clean.shape

    # Extract DWT features for each patient and each lead
    all_patient_features = []
    for patient_idx in tqdm(range(num_patients), desc="Extracting DWT features"):
        patient_row = []
        for lead_idx in range(num_leads):
            signal = X_clean[patient_idx, :, lead_idx]
            patient_row.extend(extract_single_lead_dwt(signal, wavelet=wavelet, level=level))
        all_patient_features.append(patient_row)

    # Define column names for the DWT features based on leads, coefficients, and statistics
    coeff_names = ["A", "D3", "D2", "D1"]
    stat_names = ["Variance", "Energy", "Entropy", "Max", "Min"]
    col_names = [
        f"Lead{lead_idx + 1}_{coeff_name}_{stat_name}"
        for lead_idx in range(num_leads)
        for coeff_name in coeff_names
        for stat_name in stat_names
    ]

    # Create a DataFrame from the extracted features and save it to CSV
    df_dwt = pd.DataFrame(all_patient_features, columns=col_names)
    df_dwt.to_csv(output_csv_path, index=False)
    print(f"Saved DWT features to {output_csv_path} (shape={df_dwt.shape})")
    return df_dwt

# ==============================================================================
# 4. Feature Extraction Visualization Helpers (PQRST Landmarks & DWT)
# ==============================================================================

def plot_pqrst_landmarks_slider(
    filtered_npy_path,
    patient_idx=18321,
    lead_idx=1,
    sampling_rate=100,
    window_seconds=2.0,
):
    """Interactive scrollable plot showing delineated P, Q, R, S, T landmarks."""
    X_clean = np.load(filtered_npy_path, mmap_mode="r")
    clean_signal = X_clean[patient_idx, :, lead_idx]

    del X_clean
    gc.collect()

    _, r_peaks = nk.ecg_peaks(clean_signal, sampling_rate=sampling_rate)
    _, waves = nk.ecg_delineate(
        clean_signal,
        rpeaks=r_peaks["ECG_R_Peaks"],
        sampling_rate=sampling_rate,
        method="dwt",
    )

    time_axis = np.arange(len(clean_signal)) / sampling_rate
    max_start = (len(clean_signal) / sampling_rate) - window_seconds

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(bottom=0.25, right=0.82)
    ax.plot(time_axis, clean_signal, color="black", label="ECG Signal", linewidth=1.2)
    
    def plot_landmark(key, color, label):
        indices = waves.get(key, [])
        valid_idx = [int(idx) for idx in indices if idx is not None and not np.isnan(idx)]
        if valid_idx:
            ax.scatter(
                [i / sampling_rate for i in valid_idx],
                clean_signal[valid_idx],
                color=color,
                zorder=5,
                label=label,
                s=60,
            )

    plot_landmark("ECG_P_Peaks", "blue", "P-Peak")
    plot_landmark("ECG_P_Onsets", "cyan", "P-Onset")
    plot_landmark("ECG_R_Onsets", "green", "R-Onset")
    
    r_idx = [int(idx) for idx in r_peaks.get("ECG_R_Peaks", []) if not np.isnan(idx)]
    if r_idx:
        ax.scatter(
            [i / sampling_rate for i in r_idx],
            clean_signal[r_idx],
            color="red",
            zorder=5,
            label="R-Peak",
            s=60,
        )
    
    plot_landmark("ECG_R_Offsets", "orange", "R-Offset")
    plot_landmark("ECG_T_Peaks", "purple", "T-Peak")
    plot_landmark("ECG_T_Offsets", "magenta", "T-Offset")

    ax.set_title(f"PQRST Delineation Landmarks (Patient {patient_idx}, Lead {lead_idx + 1})")
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, window_seconds)
    
    slider_ax = fig.add_axes([0.15, 0.08, 0.65, 0.04])
    start_slider = Slider(
        slider_ax, "Start (s)", 0, max_start if max_start > 0 else 1, valinit=0, valstep=0.01
    )
    
    def update(_):
        start_seconds = start_slider.val
        ax.set_xlim(start_seconds, start_seconds + window_seconds)
        fig.canvas.draw_idle()
    start_slider.on_changed(update)
    plt.show()
    return start_slider

def plot_dwt_decomposition(
    filtered_npy_path, patient_idx=14, lead_idx=0, wavelet="db4", level=3
):
    """Plot the original filtered ECG alongside its 4 DWT decomposition sub-bands."""
    X_clean = np.load(filtered_npy_path, mmap_mode="r")
    signal = X_clean[patient_idx, :, lead_idx]
    cA3, cD3, cD2, cD1 = pywt.wavedec(signal, wavelet=wavelet, level=level)
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.6)
    axes[0].plot(signal, color="black")
    axes[0].set_title(f"Filtered ECG (Patient {patient_idx}, Lead {lead_idx + 1}) - 1000 samples")
    axes[1].plot(cA3, color="blue")
    axes[1].set_title(f"cA3: Approximation (Low Frequencies / Broad Baseline) - {len(cA3)} samples")
    axes[2].plot(cD3, color="red")
    axes[2].set_title(f"cD3: Detail Level 3 (QRS Spikes) - {len(cD3)} samples")
    axes[3].plot(cD2, color="orange")
    axes[3].set_title(f"cD2: Detail Level 2 (Mid Frequencies / P & T waves) - {len(cD2)} samples")
    axes[4].plot(cD1, color="green")
    axes[4].set_title(f"cD1: Detail Level 1 (High Frequency Noise) - {len(cD1)} samples")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.6)
    plt.show()