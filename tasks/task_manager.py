import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Açıklama: Witcher karakter veri setini CSV'den okur ve bir pandas.DataFrame döndürür.
# Input:filepath: str (CSV dosya yolu)
# Output: pd.DataFrame
def load_witcher_dataset(filepath):
    return pd.read_csv(filepath)
    pass

# Açıklama: Hedef sınıf kolonunun dağılımını verir.
# Input: df: pd.DataFrame, target_column: str
# Output: dict (sınıf: adet)
def get_class_distribution(df, target_column):
    return df[target_column].value_counts().to_dict()
    pass

# Açıklama: Belirtilen kategorik kolona Label Encoding uygular.
# Input: df: pd.DataFrame, column: str
# Output: pd.DataFrame (kolon Label Encoding yapılmış)
def apply_label_encoding(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df
    pass

# Açıklama: Belirtilen kategorik kolona One Hot Encoding uygular.
# Input: df: pd.DataFrame, column: str
# Output: pd.DataFrame (kolon OHE yapılmış)
def apply_one_hot_encoding(df, column):
    return pd.get_dummies(df, columns=[column], prefix=column)
    pass

# Açıklama: Hedef kolonun dengesiz sınıflarını eşitlemek için down sampling uygular.
# Input: df: pd.DataFrame, target_column: str
# Output: pd.DataFrame (dengelenmiş)
def down_sample(df, target_column):
    siniflar = df[target_column].value_counts()
    azinlik_sayisi = siniflar.min()

    parcalar = []
    for sinif_degeri in siniflar.index:
        sinif_verisi = df[df[target_column] == sinif_degeri]
        azaltilmis = resample(sinif_verisi, n_samples=azinlik_sayisi, random_state=42)
        parcalar.append(azaltilmis)
    return pd.concat(parcalar)
    pass


# Açıklama: Hedef kolonun dengesiz sınıflarını eşitlemek için up sampling uygular.
# Input: df: pd.DataFrame, target_column: str
# Output: pd.DataFrame
def up_sample(df, target_column):
    siniflar = df[target_column].value_counts()
    cogunluk_sayisi = siniflar.max()

    parcalar = []
    for sinif_degeri in siniflar.index:
        sinif_verisi = df[df[target_column] == sinif_degeri]
        arttirilmis = resample(sinif_verisi, n_samples=cogunluk_sayisi, random_state=42)
        parcalar.append(arttirilmis)
    return pd.concat(parcalar)
    pass

# Açıklama: Girdi verisine SMOTE uygular ve yeni X, y döndürür.
# Input: X: np.ndarray veya pd.DataFrame, y: np.ndarray veya pd.Series
# Output:(X_resampled, y_resampled) (numpy array)
def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_yeni, y_yeni = smote.fit_resample(X, y)
    return X_yeni, y_yeni
    pass

# Açıklama: DataFrame'i özellikler (X) ve hedef (y) olarak ayırır.
# Input: df: pd.DataFrame, target_column: str
# Output:(X: pd.DataFrame, y: pd.Series)
def split_features_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
    pass

# Açıklama: Veri setinin temel istatistiklerini ve kolon bilgilerini döndürür.
# Input: df: pd.DataFrame
# Output: dict
def summarize_dataset(df):
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict()
    }
    pass