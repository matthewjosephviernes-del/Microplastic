# streamlit_microplastic_app.py
# Microplastic Risk Analysis Dashboard — Improved outputs & labeled visualizations
# Usage: streamlit run streamlit_microplastic_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, classification_report, mean_absolute_error, mean_squared_error
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Microplastic Risk Analysis Dashboard", page_icon="🧪")
st.title("🧪 Microplastic Risk Analysis Dashboard")
st.caption("Interactive defense demo — clearer outputs, named charts, and labeled numeric results.")

# --- Sidebar ---
st.sidebar.title("Navigation")
st.sidebar.info("Use buttons to manually trigger each stage for your thesis defense.")

st.sidebar.header("1️⃣ Upload or Load Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your .csv dataset", type=["csv"])
use_default = st.sidebar.checkbox("Use default dataset (uploaded earlier)", value=True)

def load_default(paths):
    for p in paths:
        try:
            return pd.read_csv(p)
        except Exception:
            continue
    return None

default_paths = ["/mnt/data/Cleaned_Microplastic_Dataset.csv", "/mnt/data/Encoded_Microplastic_Dataset.csv"]

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default(default_paths) if use_default else None

if df is None:
    st.warning("⚠️ Please upload a dataset or enable 'Use default dataset'.")
    st.stop()

st.success(f"✅ Dataset loaded successfully — {df.shape[0]} rows × {df.shape[1]} columns")

# --- Utility functions ---
@st.cache_data
def preprocess_dataframe(in_df):
    df = in_df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    # Keep encoders in case we want to map back later
    encoders = {}
    for c in cat_cols:
        try:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
        except Exception:
            pass

    return df, num_cols, cat_cols, encoders

@st.cache_data
def compute_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    Xp = pca.fit_transform(X)
    return Xp, pca

# --- 2️⃣ Preprocessing ---
with st.expander("⚙️ Step 2: Preprocessing", expanded=True):
    if st.button("Run Preprocessing", key="preprocess"):
        with st.spinner("Cleaning and encoding dataset..."):
            cleaned_df, num_cols, cat_cols, encoders = preprocess_dataframe(df)
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['num_cols'] = num_cols
            st.session_state['cat_cols'] = cat_cols
            st.session_state['encoders'] = encoders
            st.success("✅ Preprocessing complete!")
            st.write("Numeric columns detected:")
            st.write(num_cols)
            st.write("Categorical columns detected (encoded):")
            st.write(cat_cols)
            st.dataframe(cleaned_df.head())
    else:
        cleaned_df = st.session_state.get('cleaned_df', None)
        num_cols = st.session_state.get('num_cols', None)
        cat_cols = st.session_state.get('cat_cols', None)
        encoders = st.session_state.get('encoders', {})

if cleaned_df is None:
    st.info("Run preprocessing first.")
    st.stop()

# Ensure we have numeric columns list
numeric_columns = num_cols if num_cols else cleaned_df.select_dtypes(include=[np.number]).columns.tolist()

# --- 3️⃣ Clustering ---
with st.expander("🔹 Step 3: K-Means Clustering", expanded=False):
    default_selection = numeric_columns.copy() if numeric_columns else cleaned_df.columns.tolist()
    cluster_cols = st.multiselect("Select features for clustering (numeric recommended)", options=cleaned_df.columns.tolist(), default=default_selection)
    n_clusters = st.slider("Number of clusters", 2, 10, 3)

    if st.button("Run K-Means", key="cluster"):
        if not cluster_cols:
            st.error("Select at least one feature for clustering.")
        else:
            X = cleaned_df[cluster_cols].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_pca, pca_model = compute_pca(X_scaled)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            cleaned_df['Cluster'] = labels
            # Save scaler and kmeans for later
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['cluster_cols'] = cluster_cols
            st.session_state['cluster_scaler'] = scaler
            st.session_state['kmeans'] = kmeans
            st.success(f"✅ K-Means complete — {n_clusters} clusters identified!")

            pca_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Cluster': labels.astype(str)
            })
            # include hover text listing the selected features with their values (first few)
            hover_text = []
            for i, row in enumerate(X):
                values = ", ".join([f"{name}={val:.2f}" for name, val in zip(cluster_cols, row)])
                hover_text.append(values)
            pca_df['values'] = hover_text

            fig = px.scatter(
                pca_df, x='PC1', y='PC2', color='Cluster', hover_data=['values'],
                color_discrete_sequence=px.colors.qualitative.Vivid,
                title=f"K-Means Clusters (PCA 2D) — features: {', '.join(cluster_cols)}",
                labels={'PC1': 'PC1 (PCA component 1)', 'PC2': 'PC2 (PCA component 2)'}
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write("**Cluster Counts:**")
            cnts = cleaned_df['Cluster'].value_counts().rename_axis('Cluster').reset_index(name='Count')
            st.dataframe(cnts)

            # Display cluster centers in original feature scale
            centers_scaled = kmeans.cluster_centers_
            centers_orig = scaler.inverse_transform(centers_scaled)
            centers_df = pd.DataFrame(centers_orig, columns=cluster_cols)
            centers_df.index.name = 'Cluster'
            centers_df.reset_index(inplace=True)
            st.write("Cluster centers (original feature scale):")
            st.dataframe(centers_df)

            # Plot cluster centers as grouped bar (each cluster a facet)
            centers_melted = centers_df.melt(id_vars='Cluster', var_name='Feature', value_name='Value')
            fig_centers = px.bar(centers_melted, x='Feature', y='Value', color='Cluster', barmode='group',
                                 title="Cluster Centers by Feature (original scale)")
            st.plotly_chart(fig_centers, use_container_width=True)

# --- 4️⃣ Classification ---
with st.expander("🧠 Step 4: Random Forest Classification", expanded=False):
    # suggest likely target columns
    target_candidates = [c for c in cleaned_df.columns if c.lower() in ('risk_level','risk','label','target','risk_category','cluster')]
    # build selectbox with None + columns
    col_options = [None] + cleaned_df.columns.tolist()
    target_col = st.selectbox("Select target column (optional, defaults to Cluster)", col_options, index=0 if 'Cluster' not in cleaned_df.columns else col_options.index('Cluster'))

    if st.button("Train Classifier", key="classifier"):
        # Determine target and features
        if target_col:
            target = cleaned_df[target_col]
            X = cleaned_df.drop(columns=[target_col])
        else:
            if 'Cluster' not in cleaned_df.columns:
                st.error("No 'Cluster' found. Run clustering first or select a target column.")
                st.stop()
            target = cleaned_df['Cluster']
            X = cleaned_df.drop(columns=['Cluster'])

        X_num = X.select_dtypes(include=[np.number])
        if X_num.shape[1] == 0:
            st.error("No numeric features available for classifier after dropping the target.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_num, target, train_size=0.75, random_state=42)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)

            st.success(f"✅ Classifier trained — Accuracy: {acc:.3f}")
            st.write("Classification Accuracy (named):")
            st.write({"Classification Accuracy": f"{acc:.3f}"})

            # Confusion matrix with axis labels showing class names
            cm = confusion_matrix(y_test, preds)
            labels = np.unique(np.concatenate([y_test.astype(str), preds.astype(str)]))
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Viridis', title='Confusion Matrix',
                               labels=dict(x="Predicted label", y="True label"))
            # set tick labels if possible
            fig_cm.update_xaxes(tickvals=list(range(len(labels))), ticktext=labels.tolist())
            fig_cm.update_yaxes(tickvals=list(range(len(labels))), ticktext=labels.tolist())
            st.plotly_chart(fig_cm, use_container_width=True)

            # Detailed classification report
            try:
                report = classification_report(y_test, preds, output_dict=True)
                report_df = pd.DataFrame(report).T
                st.write("Classification report (per-class precision/recall/f1):")
                st.dataframe(report_df)
            except Exception:
                st.write("Could not produce classification report for the given labels.")

            # Feature importances with names
            importances = clf.feature_importances_
            feat_imp = pd.DataFrame({'Feature': X_num.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
            st.write("Feature importances (Random Forest):")
            st.dataframe(feat_imp)
            fig_imp = px.bar(feat_imp.head(20), x='Feature', y='Importance', title='Top Feature Importances', text='Importance')
            st.plotly_chart(fig_imp, use_container_width=True)

            # Save classifier metrics
            st.session_state['clf_acc'] = acc
            st.session_state['clf_feature_importances'] = feat_imp

# --- 5️⃣ K-Fold Validation ---
with st.expander("📊 Step 5: K-Fold Validation", expanded=False):
    n_splits = st.slider("Number of folds", 2, 10, 5)
    if st.button("Run Validation", key="validation"):
        X = cleaned_df.select_dtypes(include=[np.number])
        if 'Cluster' not in cleaned_df.columns:
            st.error("No 'Cluster' found for validation target. Run clustering first.")
        else:
            y = cleaned_df['Cluster']
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            scores = cross_val_score(clf, X, y, cv=kf)

            st.success(f"✅ Validation complete — Mean Accuracy: {scores.mean():.3f}")
            # show per-fold scores as named list/table
            folds = list(range(1, len(scores) + 1))
            scores_df = pd.DataFrame({'Fold': folds, 'Accuracy': scores})
            st.write("Per-fold accuracies:")
            st.dataframe(scores_df)

            fig = px.line(scores_df, x='Fold', y='Accuracy', markers=True, title='K-Fold Accuracy Scores', labels={'Accuracy': 'Accuracy (per fold)'})
            st.plotly_chart(fig, use_container_width=True)
            st.session_state['cv_scores'] = scores

# --- 6️⃣ Regression ---
with st.expander("📈 Step 6: Regression — Predict Risk Severity", expanded=False):
    reg_target = st.selectbox("Select numeric target for regression", [None] + cleaned_df.select_dtypes(include=[np.number]).columns.tolist())
    if st.button("Run Regression", key="regression"):
        if reg_target:
            if reg_target not in cleaned_df.columns:
                st.error("Selected regression target not in dataset.")
            else:
                y = cleaned_df[reg_target]
                X = cleaned_df.select_dtypes(include=[np.number]).drop(columns=[reg_target])
        else:
            if 'Cluster' not in cleaned_df.columns:
                st.error("No 'Cluster' found to use as regression target. Provide a numeric target or run clustering.")
                st.stop()
            y = cleaned_df['Cluster']
            X = cleaned_df.select_dtypes(include=[np.number]).drop(columns=['Cluster'])

        if X.shape[1] == 0:
            st.error("No numeric predictors available for regression.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

            regr = RandomForestRegressor(n_estimators=100, random_state=42)
            regr.fit(X_train, y_train)
            preds = regr.predict(X_test)
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            rmse = mean_squared_error(y_test, preds, squared=False)

            st.success(f"✅ Regression complete — R²: {r2:.3f}")
            st.write({"Regression R²": f"{r2:.3f}", "MAE": f"{mae:.3f}", "RMSE": f"{rmse:.3f}"})

            fig = px.scatter(x=y_test, y=preds, color_discrete_sequence=['#EF553B'],
                             title=f'Actual vs Predicted — target: {reg_target if reg_target else "Cluster"}',
                             labels={'x': f'Actual ({reg_target if reg_target else "Cluster"})', 'y': 'Predicted'})
            st.plotly_chart(fig, use_container_width=True)

            # Feature importances
            importances = regr.feature_importances_
            feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
            st.write("Regression feature importances:")
            st.dataframe(feat_imp)
            fig_imp = px.bar(feat_imp.head(20), x='Feature', y='Importance', title='Top Regression Feature Importances', text='Importance')
            st.plotly_chart(fig_imp, use_container_width=True)

            st.session_state['reg_r2'] = r2
            st.session_state['reg_metrics'] = {'r2': r2, 'mae': mae, 'rmse': rmse}

# --- 7️⃣ Final Summary ---
with st.expander("📜 Step 7: Final Summary", expanded=False):
    st.subheader("Performance Summary")
    metrics = {}
    if 'clf_acc' in st.session_state:
        metrics['Classification Accuracy'] = st.session_state['clf_acc']
    if 'cv_scores' in st.session_state:
        metrics['Mean K-Fold Accuracy'] = float(st.session_state['cv_scores'].mean())
    if 'reg_r2' in st.session_state:
        metrics['Regression R²'] = st.session_state['reg_r2']
    if 'reg_metrics' in st.session_state:
        metrics['Regression MAE'] = st.session_state['reg_metrics']['mae']
        metrics['Regression RMSE'] = st.session_state['reg_metrics']['rmse']

    if metrics:
        # Present a dataframe with named metrics for clarity
        summary_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        summary_df['Value'] = summary_df['Value'].apply(lambda x: round(float(x), 3))
        st.dataframe(summary_df)
        fig = px.bar(summary_df, x='Metric', y='Value', text='Value', color='Metric', color_discrete_sequence=px.colors.qualitative.Bold, title='Model Performance Summary (named metrics)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No metrics available yet. Run previous steps first.")

# --- Export ---
with st.expander("💾 Export Cleaned Dataset", expanded=False):
    if st.button("Generate Download Link"):
        csv = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned CSV", csv, "cleaned_microplastic_dataset.csv", "text/csv")

st.markdown("---")
st.caption("App built for thesis defense — named outputs and labeled charts help communicate numeric results clearly.")
