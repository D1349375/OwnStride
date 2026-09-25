"""
src/dataset/wisdm_loader.py
===========================
WISDM Gait Dataset Loader -- Real Human Walking Accelerometer Data
Dataset: WISDM Activity Recognition v1.1
Source: Fordham University CIS Lab (public, no registration required)
Reference: Kwapisz, J. R., Weiss, G. M., & Moore, S. A. (2011).
           Activity recognition using cell phone accelerometers.
           ACM SIGKDD Explorations Newsletter, 12(2), 74-82.

Sensor: Smartphone accelerometer (Motorola Droid, Samsung Galaxy, etc.)
Placement: Thigh (pants pocket)
Sample Rate: ~20 Hz (variable timestamp spacing)
Axes: X (lateral), Y (forward), Z (vertical) in m/s2
Subjects: 36 real human participants
Walking samples: 418,393 data points

NOTE: This is REAL measured accelerometer data from humans walking.
      It is NOT synthetic / simulated.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import urllib.request
import tarfile

WISDM_URL = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
WISDM_TARGET_RATE = 128.0   # Resample rate for the accelerometer-only cycle segmenter (event_detector.py)


class WISDMLoader:
    """
    Loads and preprocesses REAL human walking accelerometer data from the
    WISDM Activity Recognition Dataset v1.1.

    Data source: 36 participants, smartphone accelerometer, ~20 Hz.
    Walking activity subset contains 418,393 real measurements.

    Usage::

        loader = WISDMLoader(data_dir="data/raw/marea")
        loader.ensure_downloaded()
        df = loader.load_walking_data()
        acc, gyro, meta = loader.get_subject_session(subject_id=1)
    """

    RAW_FILE = "WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"

    def __init__(self, data_dir: str = "data/raw/marea", processed_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.raw_path = os.path.join(data_dir, self.RAW_FILE)
        self.processed_path = os.path.join(processed_dir, "wisdm_walking.parquet")
        self._df_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def is_downloaded(self) -> bool:
        """Return True if the raw WISDM file or its processed parquet cache is available."""
        return os.path.exists(self.processed_path) or os.path.exists(self.raw_path)

    def ensure_downloaded(self, verbose: bool = True) -> None:
        """Download and extract WISDM dataset if not already present."""
        if self.is_downloaded():
            if verbose:
                print(f"[WISDMLoader] Real data already present: {self.raw_path}")
            return

        os.makedirs(self.data_dir, exist_ok=True)
        dest = os.path.join(self.data_dir, "wisdm.tar.gz")

        if verbose:
            print("[WISDMLoader] Downloading WISDM real walking dataset (~11 MB)...")
        req = urllib.request.Request(WISDM_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(dest, "wb") as f:
                f.write(resp.read())

        if verbose:
            size_mb = os.path.getsize(dest) / 1e6
            print(f"[WISDMLoader] Downloaded {size_mb:.1f} MB. Extracting...")

        with tarfile.open(dest, "r:gz") as t:
            t.extractall(self.data_dir)

        if verbose:
            print(f"[WISDMLoader] Real WISDM data ready at {self.raw_path}")

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_raw(self) -> pd.DataFrame:
        """Parse WISDM raw text file into a DataFrame."""
        rows = []
        with open(self.raw_path, "r") as f:
            for line in f:
                line = line.strip().rstrip(";")
                parts = line.split(",")
                if len(parts) != 6:
                    continue
                try:
                    user     = int(parts[0])
                    activity = parts[1].strip()
                    ts       = int(parts[2])
                    x        = float(parts[3])
                    y        = float(parts[4])
                    z        = float(parts[5])
                    rows.append((user, activity, ts, x, y, z))
                except (ValueError, IndexError):
                    continue

        df = pd.DataFrame(rows, columns=["user", "activity", "timestamp", "acc_x", "acc_y", "acc_z"])
        # Convert from m/s2 to g (WISDM uses m/s2, our pipeline uses g)
        for col in ["acc_x", "acc_y", "acc_z"]:
            df[col] = df[col] / 9.81
        return df

    def load_walking_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Return a DataFrame of REAL walking accelerometer data.

        Returns
        -------
        pd.DataFrame with columns: user, activity, timestamp, acc_x, acc_y, acc_z
        All acc values are in g (divided by 9.81 from original m/s2).
        Source: 36 real human participants, 418,393 walking samples.
        """
        if not self.is_downloaded():
            raise FileNotFoundError(
                f"WISDM data not found at {self.raw_path}. "
                "Call ensure_downloaded() first."
            )

        if use_cache and self._df_cache is not None:
            return self._df_cache[self._df_cache["activity"] == "Walking"].copy()

        # Check parquet fast cache
        if use_cache and os.path.exists(self.processed_path):
            df = pd.read_parquet(self.processed_path)
            self._df_cache = df
            return df[df["activity"] == "Walking"].copy()

        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(
                f"WISDM data not found at {self.raw_path}. "
                "Call ensure_downloaded() first."
            )

        df = self._parse_raw()
        try:
            os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
            df.to_parquet(self.processed_path, index=False)
        except Exception:
            pass
        self._df_cache = df
        return df[df["activity"] == "Walking"].copy()

    # ------------------------------------------------------------------
    # Session extraction (compatible with SyntheticGaitStream interface)
    # ------------------------------------------------------------------

    def get_subject_session(
        self,
        subject_id: int = 1,
        n_samples: Optional[int] = None,
        resample_to_hz: float = WISDM_TARGET_RATE,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Extract a walking session for a specific subject and resample to
        the target sample rate (default 128 Hz) to match the pipeline.

        Parameters
        ----------
        subject_id : int
            Logical subject index 1-N (mapped to WISDM user IDs 1-36).
        n_samples : int, optional
            Maximum number of samples to return after resampling.
        resample_to_hz : float
            Target sample rate in Hz (default 128.0 to match pipeline).

        Returns
        -------
        (acc, gyro_zeros, metadata)
        - acc      : shape (N, 3), float64, in g -- REAL accelerometer data
        - gyro     : shape (N, 3), float64 -- zeros (WISDM has no gyroscope)
        - metadata : dict with provenance information

        NOTE: gyro is all zeros because WISDM only records accelerometer.
              Do NOT fill gyro with synthetic data -- that would defeat the
              purpose of using a real dataset.
        """
        walking_df = self.load_walking_data()

        available_users = sorted(walking_df["user"].unique())
        if not available_users:
            raise ValueError("No walking data found in WISDM dataset.")

        idx = (subject_id - 1) % len(available_users)
        actual_user = available_users[idx]

        subj_df = walking_df[walking_df["user"] == actual_user].copy()
        subj_df = subj_df.sort_values("timestamp").reset_index(drop=True)

        original_n = len(subj_df)
        original_hz = 20.0

        acc_raw = subj_df[["acc_x", "acc_y", "acc_z"]].values

        if resample_to_hz != original_hz:
            from scipy.interpolate import interp1d
            t_orig = np.linspace(0, original_n / original_hz, original_n)
            duration = original_n / original_hz
            t_new = np.arange(0, duration, 1.0 / resample_to_hz)
            acc_resampled = np.zeros((len(t_new), 3))
            for ch in range(3):
                interp_fn = interp1d(t_orig, acc_raw[:, ch], kind="linear",
                                     fill_value="extrapolate")
                acc_resampled[:, ch] = interp_fn(t_new)
        else:
            acc_resampled = acc_raw.copy()

        if n_samples is not None:
            acc_resampled = acc_resampled[:n_samples]

        gyro_zeros = np.zeros_like(acc_resampled)

        metadata = {
            "source": "WISDM_real_human_walking",
            "dataset": "WISDM Activity Recognition v1.1",
            "citation": "Kwapisz et al. (2011), ACM SIGKDD Explorations, 12(2), 74-82",
            "url": "https://www.cis.fordham.edu/wisdm/dataset.php",
            "wisdm_user_id": int(actual_user),
            "requested_subject_id": subject_id,
            "original_samples": original_n,
            "original_hz": original_hz,
            "resampled_hz": resample_to_hz,
            "resampled_samples": len(acc_resampled),
            "gyro_available": False,
            "is_synthetic": False,      # Explicit provenance flag
        }

        return acc_resampled, gyro_zeros, metadata

    def get_all_subjects_summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of walking data per subject.
        Useful for reporting dataset statistics in the proposal.
        """
        walking_df = self.load_walking_data()
        rows = []
        for user, grp in walking_df.groupby("user"):
            mag = np.sqrt(grp["acc_x"]**2 + grp["acc_y"]**2 + grp["acc_z"]**2)
            rows.append({
                "user": user,
                "n_samples": len(grp),
                "acc_x_mean": grp["acc_x"].mean(),
                "acc_y_mean": grp["acc_y"].mean(),
                "acc_z_mean": grp["acc_z"].mean(),
                "magnitude_std": mag.std(),
            })
        return pd.DataFrame(rows)
