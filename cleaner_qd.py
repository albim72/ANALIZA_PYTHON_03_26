import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import nan_euclidean_distances
from collections import Counter
import re
import warnings

warnings.filterwarnings("ignore")


# =========================================================
# POMOCNICZE STRUKTURY
# =========================================================

@dataclass
class CellDecision:
    row_idx: int
    col_name: str
    original_value: Any
    cleaned_value: Any
    confidence: float
    reason: str
    candidates: List[Tuple[Any, float]] = field(default_factory=list)


@dataclass
class FeatureProfile:
    name: str
    inferred_type: str
    missing_ratio: float
    unique_ratio: float
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Any] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    top_values: Optional[List[Any]] = None


# =========================================================
# GŁÓWNY SILNIK QUANTUM DICE CLEANER
# =========================================================

class QuantumDiceCleaner:
    def __init__(
        self,
        contamination: float = 0.03,
        similarity_k: int = 5,
        confidence_threshold: float = 0.55,
        random_state: int = 42
    ):
        self.contamination = contamination
        self.similarity_k = similarity_k
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state

        self.feature_profiles: Dict[str, FeatureProfile] = {}
        self.decision_log: List[CellDecision] = []
        self.label_encoders: Dict[str, LabelEncoder] = {}

    # =====================================================
    # 1. PROFILOWANIE CECH
    # =====================================================
    def infer_feature_type(self, s: pd.Series) -> str:
        non_null = s.dropna()

        if len(non_null) == 0:
            return "unknown"

        # próba liczby
        numeric_try = pd.to_numeric(non_null.astype(str).str.replace(",", "."), errors="coerce")
        numeric_ratio = numeric_try.notna().mean()

        if numeric_ratio > 0.9:
            return "numeric"

        # próba daty
        datetime_try = pd.to_datetime(non_null, errors="coerce")
        datetime_ratio = datetime_try.notna().mean()

        if datetime_ratio > 0.9:
            return "datetime"

        # tekst / kategoria
        nunique = non_null.nunique()
        unique_ratio = nunique / max(len(non_null), 1)

        if unique_ratio < 0.3:
            return "categorical"

        return "text"

    def profile_features(self, df: pd.DataFrame) -> Dict[str, FeatureProfile]:
        profiles = {}

        for col in df.columns:
            s = df[col]
            inferred = self.infer_feature_type(s)
            non_null = s.dropna()

            profile = FeatureProfile(
                name=col,
                inferred_type=inferred,
                missing_ratio=s.isna().mean(),
                unique_ratio=non_null.nunique() / max(len(non_null), 1)
            )

            if inferred == "numeric":
                numeric = pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")
                profile.mean = float(numeric.mean()) if numeric.notna().any() else None
                profile.std = float(numeric.std()) if numeric.notna().any() else None
                profile.median = float(numeric.median()) if numeric.notna().any() else None
                profile.min_val = float(numeric.min()) if numeric.notna().any() else None
                profile.max_val = float(numeric.max()) if numeric.notna().any() else None

            elif inferred in ["categorical", "text"]:
                mode_vals = non_null.mode()
                profile.mode = mode_vals.iloc[0] if len(mode_vals) > 0 else None
                profile.top_values = list(non_null.astype(str).value_counts().head(10).index)

            profiles[col] = profile

        self.feature_profiles = profiles
        return profiles

    # =====================================================
    # 2. NORMALIZACJA WSTĘPNA
    # =====================================================
    def normalize_value(self, value: Any, feature_type: str) -> Any:
        if pd.isna(value):
            return np.nan

        if feature_type == "numeric":
            if isinstance(value, str):
                value = value.strip().replace(",", ".")
            try:
                return float(value)
            except Exception:
                return np.nan

        if feature_type == "categorical":
            if isinstance(value, str):
                value = value.strip().lower()
                value = re.sub(r"\s+", " ", value)
            return value

        if feature_type == "text":
            if isinstance(value, str):
                value = value.strip()
                value = re.sub(r"\s+", " ", value)
            return value

        if feature_type == "datetime":
            try:
                return pd.to_datetime(value, errors="coerce")
            except Exception:
                return pd.NaT

        return value

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_norm = df.copy()

        for col, profile in self.feature_profiles.items():
            df_norm[col] = df_norm[col].apply(lambda x: self.normalize_value(x, profile.inferred_type))

        return df_norm

    # =====================================================
    # 3. ENKODOWANIE DLA ANALIZY KONTEKSTU
    # =====================================================
    def encode_for_context(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.DataFrame(index=df.index)

        for col, profile in self.feature_profiles.items():
            s = df[col]

            if profile.inferred_type == "numeric":
                encoded[col] = pd.to_numeric(s, errors="coerce")

            elif profile.inferred_type == "datetime":
                dt = pd.to_datetime(s, errors="coerce")
                encoded[col] = dt.astype("int64") / 10**9
                encoded[col] = encoded[col].replace(-9.223372e+09, np.nan)

            else:
                le = LabelEncoder()
                s_str = s.astype(str).fillna("__MISSING__")
                le.fit(s_str)
                encoded[col] = le.transform(s_str).astype(float)
                self.label_encoders[col] = le

        return encoded

    # =====================================================
    # 4. WYKRYWANIE ANOMALII GLOBALNYCH
# =====================================================
    def detect_row_anomalies(self, encoded_df: pd.DataFrame) -> pd.Series:
        temp = encoded_df.copy()

        # proste uzupełnienie tylko do detekcji anomalii
        for col in temp.columns:
            median = temp[col].median()
            temp[col] = temp[col].fillna(median)

        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        preds = model.fit_predict(temp)
        return pd.Series(preds == -1, index=temp.index)

    # =====================================================
    # 5. PODOBIEŃSTWO REKORDÓW
    # =====================================================
    def get_nearest_neighbors(self, encoded_df: pd.DataFrame, row_idx: int, k: int = 5) -> List[int]:
        distances = nan_euclidean_distances(encoded_df)
        row_pos = encoded_df.index.get_loc(row_idx)
        row_dist = distances[row_pos]

        neighbors = np.argsort(row_dist)
        neighbors = [encoded_df.index[i] for i in neighbors if encoded_df.index[i] != row_idx]

        return neighbors[:k]

    # =====================================================
    # 6. GENEROWANIE KANDYDATÓW NAPRAWY
    # =====================================================
    def generate_candidates(
        self,
        df: pd.DataFrame,
        encoded_df: pd.DataFrame,
        row_idx: int,
        col: str
    ) -> List[Any]:
        profile = self.feature_profiles[col]
        value = df.loc[row_idx, col]
        candidates = []

        if profile.inferred_type == "numeric":
            candidates.append(profile.median)
            candidates.append(profile.mean)

            neighbors = self.get_nearest_neighbors(encoded_df, row_idx, self.similarity_k)
            neigh_vals = pd.to_numeric(df.loc[neighbors, col], errors="coerce").dropna()
            if len(neigh_vals) > 0:
                candidates.append(float(neigh_vals.median()))
                candidates.append(float(neigh_vals.mean()))

        elif profile.inferred_type == "categorical":
            if profile.mode is not None:
                candidates.append(profile.mode)

            neighbors = self.get_nearest_neighbors(encoded_df, row_idx, self.similarity_k)
            neigh_vals = df.loc[neighbors, col].dropna().astype(str).tolist()
            if neigh_vals:
                candidates.append(Counter(neigh_vals).most_common(1)[0][0])

            if profile.top_values:
                candidates.extend(profile.top_values[:3])

        elif profile.inferred_type == "datetime":
            candidates.append(df[col].dropna().mode().iloc[0] if not df[col].dropna().empty else pd.NaT)

        elif profile.inferred_type == "text":
            if profile.mode is not None:
                candidates.append(profile.mode)

        # usuwanie duplikatów
        uniq = []
        seen = set()
        for c in candidates:
            key = str(c)
            if key not in seen and pd.notna(c):
                seen.add(key)
                uniq.append(c)

        return uniq

    # =====================================================
    # 7. QUANTUM DICE SCORING
    # =====================================================
    def quantum_dice_score(
        self,
        df: pd.DataFrame,
        encoded_df: pd.DataFrame,
        row_idx: int,
        col: str,
        candidate: Any
    ) -> float:
        """
        Tutaj jest miejsce na Twój właściwy algorytm Quantum Dice.
        Na razie używam wersji bazowej:
        - zgodność z profilem kolumny
        - zgodność z sąsiadami
        - kara za dużą zmianę
        """

        profile = self.feature_profiles[col]
        original = df.loc[row_idx, col]

        feature_score = 0.0
        neighbor_score = 0.0
        penalty = 0.0

        if profile.inferred_type == "numeric":
            cand = pd.to_numeric(pd.Series([candidate]), errors="coerce").iloc[0]
            if pd.notna(cand) and profile.mean is not None and profile.std is not None:
                z = abs(cand - profile.mean) / (profile.std + 1e-8)
                feature_score = max(0.0, 1.0 - z / 5.0)

            neighbors = self.get_nearest_neighbors(encoded_df, row_idx, self.similarity_k)
            neigh_vals = pd.to_numeric(df.loc[neighbors, col], errors="coerce").dropna()
            if len(neigh_vals) > 0:
                local_mean = neigh_vals.mean()
                local_std = neigh_vals.std() + 1e-8
                z_local = abs(cand - local_mean) / local_std if local_std > 0 else 0
                neighbor_score = max(0.0, 1.0 - z_local / 5.0)

            if pd.notna(original):
                try:
                    penalty = min(1.0, abs(float(original) - float(cand)) / (abs(float(original)) + 1e-8))
                except Exception:
                    penalty = 0.2

        elif profile.inferred_type == "categorical":
            feature_score = 1.0 if str(candidate) in map(str, profile.top_values or []) else 0.4

            neighbors = self.get_nearest_neighbors(encoded_df, row_idx, self.similarity_k)
            neigh_vals = df.loc[neighbors, col].dropna().astype(str).tolist()
            if neigh_vals:
                neighbor_score = neigh_vals.count(str(candidate)) / len(neigh_vals)

            penalty = 0.0 if pd.isna(original) else 0.25

        elif profile.inferred_type == "text":
            feature_score = 0.6
            neighbor_score = 0.4
            penalty = 0.1 if pd.notna(original) else 0.0

        elif profile.inferred_type == "datetime":
            feature_score = 0.7
            neighbor_score = 0.5
            penalty = 0.1 if pd.notna(original) else 0.0

        score = (
            0.45 * feature_score +
            0.35 * neighbor_score -
            0.20 * penalty
        )

        return float(np.clip(score, 0.0, 1.0))

    # =====================================================
    # 8. CZY WARTO CZYŚCIĆ DANĄ KOMÓRKĘ?
    # =====================================================
    def cell_needs_cleaning(self, df: pd.DataFrame, row_idx: int, col: str) -> bool:
        value = df.loc[row_idx, col]
        profile = self.feature_profiles[col]

        if pd.isna(value):
            return True

        if profile.inferred_type == "numeric":
            num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(num):
                return True

            if profile.mean is not None and profile.std is not None and profile.std > 0:
                z = abs(num - profile.mean) / profile.std
                if z > 4:
                    return True

        if profile.inferred_type == "categorical":
            val = str(value)
            top = profile.top_values or []
            if val not in map(str, top) and profile.unique_ratio < 0.2:
                return True

        return False

    # =====================================================
    # 9. CZYSZCZENIE JEDNEJ KOMÓRKI
    # =====================================================
    def clean_cell(
        self,
        df: pd.DataFrame,
        encoded_df: pd.DataFrame,
        row_idx: int,
        col: str
    ) -> Tuple[Any, float, str, List[Tuple[Any, float]]]:
        original = df.loc[row_idx, col]
        candidates = self.generate_candidates(df, encoded_df, row_idx, col)

        if not candidates:
            return original, 0.0, "Brak kandydatów naprawy", []

        scored = []
        for cand in candidates:
            score = self.quantum_dice_score(df, encoded_df, row_idx, col, cand)
            scored.append((cand, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_value, best_score = scored[0]

        if best_score >= self.confidence_threshold:
            return best_value, best_score, "Wybrano najlepszy kandydat przez Quantum Dice", scored

        return original, best_score, "Nie osiągnięto progu pewności", scored

    # =====================================================
    # 10. GŁÓWNE CZYSZCZENIE ZBIORU
    # =====================================================
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.decision_log = []

        # profilowanie
        self.profile_features(df)

        # normalizacja
        clean_df = self.normalize_dataframe(df)

        # kontekst
        encoded_df = self.encode_for_context(clean_df)

        # anomalie wierszowe
        anomaly_rows = self.detect_row_anomalies(encoded_df)

        for row_idx in clean_df.index:
            for col in clean_df.columns:
                needs_clean = self.cell_needs_cleaning(clean_df, row_idx, col)

                # jeśli cały wiersz jest anomalią, zwiększamy czujność
                if anomaly_rows.loc[row_idx]:
                    needs_clean = needs_clean or pd.isna(clean_df.loc[row_idx, col])

                if needs_clean:
                    new_value, conf, reason, candidates = self.clean_cell(
                        clean_df, encoded_df, row_idx, col
                    )

                    if new_value is not clean_df.loc[row_idx, col] or pd.isna(clean_df.loc[row_idx, col]):
                        self.decision_log.append(
                            CellDecision(
                                row_idx=row_idx,
                                col_name=col,
                                original_value=clean_df.loc[row_idx, col],
                                cleaned_value=new_value,
                                confidence=conf,
                                reason=reason,
                                candidates=candidates
                            )
                        )
                        clean_df.loc[row_idx, col] = new_value

        return clean_df

    # =====================================================
    # 11. RAPORT DECYZJI
    # =====================================================
    def get_decision_report(self) -> pd.DataFrame:
        rows = []
        for d in self.decision_log:
            rows.append({
                "row_idx": d.row_idx,
                "col_name": d.col_name,
                "original_value": d.original_value,
                "cleaned_value": d.cleaned_value,
                "confidence": d.confidence,
                "reason": d.reason,
                "candidates": d.candidates
            })
        return pd.DataFrame(rows)


# ==========================================
# PRZYKŁADOWE DANE
# ==========================================
data = {
    "age": [25, 27, "29", None, 300, 31, "xx", 28],
    "city": ["Warsaw", "warsaw", "Krakow", None, "Warsaw", "krakow", "Wroclaw", "Warsaw"],
    "salary": [5000, 5200, None, 5100, 9999999, 5300, 5250, "5100"],
    "department": ["IT", "it", "HR", "HR", "IT", None, "Finance", "IT"]
}

df = pd.DataFrame(data)

cleaner = QuantumDiceCleaner(
    contamination=0.1,
    similarity_k=3,
    confidence_threshold=0.5
)

clean_df = cleaner.fit_transform(df)
report_df = cleaner.get_decision_report()

print("=== DANE ORYGINALNE ===")
print(df)
print("\n=== DANE PO CZYSZCZENIU ===")
print(clean_df)
print("\n=== RAPORT DECYZJI ===")
print(report_df)
