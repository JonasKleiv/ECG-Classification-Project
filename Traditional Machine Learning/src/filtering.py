import gc
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import Slider
import neurokit2 as nk
import numpy as np
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

def apply_butterworth_filter(
        input_cache_path,
        output_cache_path,
        order,
        cutoff_frequency,
        sampling_rate,
        btype="highpass",
        force_reload=False,
        
):
    """
    Apply a zero-phase Butterworth filter to cached ECG signals and save to disk.
    Skips computation if the filtered cache file already exists and force_reload is False.
    """
    # Skip filtering if the output cache already exists and force_reload is False
    if os.path.exists(output_cache_path) and not force_reload:
        print(f"Filtered cache already exists at: {output_cache_path}")
        return

    print(f"Applying {order}th-order Butterworth {btype} filter ({cutoff_frequency} Hz)...")

    # Load the raw ECG signals from the input cache
    X_raw = np.load(input_cache_path)

    # Design the Butterworth filter in second-order sections (SOS) format
    sos = butter(order, cutoff_frequency, btype=btype, fs=sampling_rate, output='sos')

    # Apply the zero-phase Butterworth filter to the raw ECG signals
    X_filtered = sosfiltfilt(sos, X_raw, axis=1)
    np.save(output_cache_path, X_filtered)

    # Clean up memory
    del X_raw
    del X_filtered
    gc.collect()

def _clean_single_record_nk(record, sampling_rate):
    """
    Helper to clean all 12 leads of an ECG record using NeuroKit2.
    """
    # Initialize an array to hold the cleaned ECG signals
    clean_record = np.zeros_like(record)

    # Clean each lead of the ECG record individually
    for lead in range(record.shape[1]):
        clean_record[:, lead] = nk.ecg_clean(record[:, lead], sampling_rate=sampling_rate)
    return clean_record

def apply_nk_clean_filter(
        input_cache_path,
        output_cache_path,
        sampling_rate,
        force_reload=False,
):
    """
    Apply NeuroKit2 cleaning to cached ECG signals and save to disk.
    Skips computation if the cleaned cache file already exists and force_reload is False.
    """
    # Skip cleaning if the output cache already exists and force_reload is False
    if os.path.exists(output_cache_path) and not force_reload:
        print(f"NeuroKit2 cleaned cache already exists at: {output_cache_path}")
        return

    # Load the raw ECG signals from the input cache
    X_raw = np.load(input_cache_path)

    # Apply NeuroKit2 cleaning to each ECG record in parallel with a progress bar
    results = list(
        tqdm(
            Parallel(return_as="generator", n_jobs=-1)(
                delayed(_clean_single_record_nk)(record, sampling_rate)
                for record in X_raw
            ),
            total=len(X_raw),
            desc="Cleaning ECG Records (NeuroKit2)",
        )
    )

    # Convert the list of cleaned records to a NumPy array
    X_filtered = np.array(results)

    np.save(output_cache_path, X_filtered)
    print(f"Saved NeuroKit2 cleaned signals to: {output_cache_path}")
    
    # Clean up memory
    del X_raw
    del X_filtered
    del results
    gc.collect()

def plot_ecg_comparison_slider(
    raw_cache_path,
    filtered_cache_path,
    Y_df,
    record_index=18293,
    lead=1,
    sampling_rate=100,
    window_seconds=3.0,
):
    """
    Launch an interactive clinical ECG grid viewer comparing raw vs. filtered signals
    with a time-window slider.
    """

    # Load the raw and filtered ECG signals from the cache files
    X_raw_cache = np.load(raw_cache_path, mmap_mode="r")
    X_filtered_cache = np.load(filtered_cache_path, mmap_mode="r")

    # Extract the raw and filtered signals for the specified record and lead (.copy() releases mmap lock)
    raw_signal = X_raw_cache[record_index, :, lead].copy()
    filtered_signal = X_filtered_cache[record_index, :, lead].copy()
    del X_raw_cache, X_filtered_cache

    # Calculate the duration of the signal and adjust the window size if necessary
    duration = len(raw_signal) / sampling_rate
    window_seconds = min(window_seconds, duration)
    max_start = max(0, duration - window_seconds)

    # Determine the y-axis limit for plotting based on the maximum absolute value of the signals
    y_limit = max(
        1,
        np.ceil(
            max(np.max(np.abs(raw_signal)), np.max(np.abs(filtered_signal))) * 2
        ) / 2,
    )

    # Configure the plots
    fig, (ax_raw, ax_filtered) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.subplots_adjust(bottom=0.25, hspace=0.4)
    (raw_line,) = ax_raw.plot([], [], color="black", linewidth=0.8)
    (filtered_line,) = ax_filtered.plot([], [], color="black", linewidth=0.8)
    for ax in (ax_raw, ax_filtered):
        # Standard ECG paper grid: small boxes (0.04s x 0.1mV), large boxes (0.2s x 0.5mV)
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.grid(which="minor", color="#f3b6b6", linewidth=0.5)
        ax.grid(which="major", color="#d66", linewidth=0.9)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xlim(0, window_seconds)
        ax.set_aspect(0.2, adjustable="box")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)")

    # Set the titles for the raw and filtered ECG plots
    record_name = Y_df.index[record_index] if Y_df is not None else record_index
    ax_raw.set_title(f"Raw ECG Signal - Record {record_name}, Lead {lead + 1}")
    ax_filtered.set_title(f"Filtered ECG Signal - Record {record_name}, Lead {lead + 1}")

    # Create a slider to control the start time of the displayed window
    slider_ax = fig.add_axes([0.15, 0.08, 0.7, 0.04])
    start_slider = Slider(
        slider_ax,
        "Start (s)",
        0,
        max_start if max_start > 0 else 1,
        valinit=0,
        valstep=0.01,
    )

    # Define the update function for the slider
    def update(start_seconds):
        start = int(start_seconds * sampling_rate)
        end = min(int((start_seconds + window_seconds) * sampling_rate), len(raw_signal))
        time = np.arange(start, end) / sampling_rate
        raw_line.set_data(time, raw_signal[start:end])
        filtered_line.set_data(time, filtered_signal[start:end])
        ax_raw.set_xlim(start_seconds, start_seconds + window_seconds)
        fig.canvas.draw_idle()
    
    start_slider.on_changed(update)
    update(0)
    plt.show()

    # Returning slider keeps widget reference alive in Jupyter
    return start_slider  

