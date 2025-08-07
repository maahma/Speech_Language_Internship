import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

features_path = "/users/ac4ma/Speech_Language_Internship/feature_extraction/features_speech_emobase"
plots_folder = "/users/ac4ma/Speech_Language_Internship/feature_extraction/plots_speech_features"
os.makedirs(plots_folder, exist_ok=True)

def plot_feature_with_ci(x, y, feature_name, ylabel, output_path):
    y = np.array(y)
    y_smooth = gaussian_filter1d(y, sigma=2)

    window_size = 5
    std_dev = pd.Series(y).rolling(window=window_size, min_periods=1, center=True).std()
    std_dev_smooth = gaussian_filter1d(std_dev.values, sigma=2)

    upper = y_smooth + std_dev_smooth
    lower = y_smooth - std_dev_smooth

    plt.figure(figsize=(12, 5))
    plt.plot(x, y_smooth, color='blue', label=f"{feature_name} (Mean)")
    plt.fill_between(x, lower, upper, color='blue', alpha=0.2, label='±1 Std Dev')
    plt.title(f"{feature_name} with Confidence Interval")
    plt.xlabel("Segment Number")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


for company in os.listdir(features_path):
    company_path = os.path.join(features_path, company)

    logging.info(f"Processing {company}")

    segment_ids = []

    feature_data = {
        "F0_sma_amean": [],
        "pcm_intensity_sma_amean": [],
    }

    # feature_data = {
#     "F0_sma_amean": [],
#     "F0_sma_stddev": [],
#     "pcm_intensity_sma_amean": [],
#     "pcm_intensity_sma_stddev": [],
#     "mfcc_sma[1]_amean": [],
# }


    for file in sorted(os.listdir(company_path)):
        file_path = os.path.join(company_path, file)
        df = pd.read_csv(file_path)
        seg_id = int(os.path.splitext(file)[0])
        segment_ids.append(seg_id)

        for feat in feature_data:
            feature_data[feat].append(df[feat].values[0])

    output_company_dir = os.path.join(plots_folder, company)
    os.makedirs(output_company_dir, exist_ok=True)

    plot_feature_with_ci(
        segment_ids,
        feature_data["F0_sma_amean"],
        "F0_sma_amean",
        "Average Pitch (Hz)",
        os.path.join(output_company_dir, "pitch_with_ci.png"),
    )

    plot_feature_with_ci(
        segment_ids,
        feature_data["pcm_intensity_sma_amean"],
        "pcm_intensity_sma_amean",
        "Average Intensity",
        os.path.join(output_company_dir, "intensity_with_ci.png"),
    )

    logging.info(f"Saved plots for {company}")