import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
import joblib

# Veri setini oku
df = pd.read_csv("winequality-red.csv", sep=';')

# Quality değerini sınıflara ayır: 0 = Kötü, 1 = Orta, 2 = İyi
def kalite_sinifla(q):
    if q <= 5:
        return 0
    elif q == 6:
        return 1
    else:
        return 2

df['quality_label'] = df['quality'].apply(kalite_sinifla)

# Kullanılacak özellikler
features = ['alcohol', 'sulphates', 'density', 'total sulfur dioxide']
X = df[features]
y = df['quality_label']

# Eğitim/test ayırımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Ölçekleme
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LightGBM modelini eğit
model = LGBMClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Test seti tahmini
y_pred = model.predict(X_test_scaled)

# Performans değerlendirmesi
print("Doğruluk:", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred, target_names=["Kötü", "Orta", "İyi"]))

# Model ve scaler'ı kaydet
joblib.dump(model, "wine_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ LightGBM modeli ve scaler kaydedildi.")
