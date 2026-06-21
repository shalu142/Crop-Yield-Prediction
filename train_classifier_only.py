import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print('Loading data.csv')
rec_df = pd.read_csv('data.csv')
required_columns = ['N','P','K','temperature','humidity','ph','rainfall','label']
if not all(col in rec_df.columns for col in required_columns):
    raise ValueError('data.csv missing required columns')
rec_df = rec_df.dropna(subset=required_columns)
rec_df['NPK_RATIO'] = rec_df['N'] / (rec_df['P'] + rec_df['K'] + 1e-6)
rec_df['WEATHER_INDEX'] = rec_df['temperature'] * rec_df['humidity'] / 100 + rec_df['rainfall'] / 100
features = ['N','P','K','temperature','humidity','ph','rainfall','NPK_RATIO','WEATHER_INDEX']
X = rec_df[features]
y = rec_df['label']
le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print('Training classifier on', X_train.shape[0], 'rows,', len(le.classes_), 'classes')
pipeline = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
pipeline.fit(X_train, y_train)
print('Predicting...')
y_pred = pipeline.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
joblib.dump(pipeline, 'crop_classification_model.pkl')
joblib.dump(le, 'crop_label_encoder.pkl')
print('Saved crop_classification_model.pkl and crop_label_encoder.pkl')
