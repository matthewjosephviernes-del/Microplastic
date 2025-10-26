# streamlit_microplastic_app.py
# Microplastic Risk Analysis Dashboard — Enhanced clarity & interpretability
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

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Microplastic Risk Analysis Dashboard", page_icon="🧪")
st.title("🧪 Microplastic Risk Analysis Dashboard")
st.caption("Now with clearer summaries, named results, cluster severity mapping, and example prediction tables for easy interpretation.")

# --- Sidebar ---
st.sidebar.title("Navigation")
st.sidebar.info("Run each step to produce human-readable results. Use the 'Reference column' to create intuitive cluster severity labels.")

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

st.success(f"✅ Dataset loaded — {df.shape[0]} rows × {df.shape[1]} columns")

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

def color_for_score(score, higher_is_better=True):
    # return 'green','yellow','red' based on thresholds for display purposes
    if higher_is_better:
        if score >= 0.85:
            return "🟢"
        elif score >= 0.6:
            return "🟡"
        else:
            return "🔴"
    else:
        # lower better (like MAE/RMSE)
        if score <= 0.1:
            return "🟢"
        elif score <= 0.5:
            return "🟡"
        else:
            return "🔴"

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
            st.markdown("**Detected numeric columns:**")
            st.write(num_cols)
            st.markdown("**Detected categorical columns (encoded):**")
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

numeric_columns = num_cols if num_cols else cleaned_df.select_dtypes(include=[np.number]).columns.tolist()

# --- 3️⃣ Clustering ---
with st.expander("🔹 Step 3: K-Means Clustering & Cluster Severity", expanded=False):
    default_selection = numeric_columns.copy() if numeric_columns else cleaned_df.columns.tolist()
    cluster_cols = st.multiselect("Select features for clustering (numeric recommended)", options=cleaned_df.columns.tolist(), default=default_selection)
    n_clusters = st.slider("Number of clusters", 2, 10, 3)
    severity_ref = st.selectbox("Reference numeric column to rank cluster severity (used to label clusters 'Low/Medium/High')", options=numeric_columns, index=0 if numeric_columns else None)

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
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['cluster_cols'] = cluster_cols
            st.session_state['cluster_scaler'] = scaler
            st.session_state['kmeans'] = kmeans

            # PCA scatter with readable hover text (feature=value)
            pca_df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Cluster': labels.astype(str)})
            hover_text = []
            for row in X:
                values = ", ".join([f"{name}={val:.2f}" for name, val in zip(cluster_cols, row)])
                hover_text.append(values)
            pca_df['values'] = hover_text
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', hover_data=['values'],
                             title=f"K-Means Clusters (PCA 2D) — features: {', '.join(cluster_cols)}",
                             labels={'PC1': 'PC1 (PCA comp 1)', 'PC2': 'PC2 (PCA comp 2)'})
            st.plotly_chart(fig, use_container_width=True)

            # Cluster counts
            cnts = cleaned_df['Cluster'].value_counts().rename_axis('Cluster').reset_index(name='Count')
            st.write("**Cluster counts:**")
            st.dataframe(cnts)

            # Cluster centers back in original scale
            centers_scaled = kmeans.cluster_centers_
            centers_orig = scaler.inverse_transform(centers_scaled)
            centers_df = pd.DataFrame(centers_orig, columns=cluster_cols)
            centers_df.index.name = 'Cluster'
            centers_df.reset_index(inplace=True)
            st.write("Cluster centers (original scale):")
            st.dataframe(centers_df.round(3))

            # Create an interpretable severity ranking based on reference column means
            if severity_ref:
                cluster_means = cleaned_df.groupby('Cluster')[severity_ref].mean().rename('MeanRef').reset_index()
                # rank clusters by mean value
                cluster_means = cluster_means.sort_values('MeanRef').reset_index(drop=True)
                # assign severity labels by quantiles / number of clusters
                labels_map = {}
                n = len(cluster_means)
                # create a severity palette based on n
                if n <= 3:
                    severity_levels = ['Low', 'Medium', 'High'][:n]
                elif n == 4:
                    severity_levels = ['Very Low', 'Low', 'High', 'Very High']
                else:
                    # create levels by quantile buckets
                    severity_levels = []
                    for i in range(n):
                        severity_levels.append(f"Level {i+1}")
                cluster_means['Severity'] = severity_levels[:n]
                # build mapping from cluster id to severity by sorting on MeanRef
                sorted_clusters = cluster_means['Cluster'].tolist()
                mapping = dict(zip(sorted_clusters, severity_levels[:n]))
                # apply mapping to cleaned_df
                cleaned_df['ClusterSeverity'] = cleaned_df['Cluster'].map(mapping)
                st.session_state['cleaned_df'] = cleaned_df
                # show cluster summary with severity and mean
                cluster_summary = cleaned_df.groupby(['Cluster', 'ClusterSeverity'])[severity_ref].agg(['count','mean']).reset_index().rename(columns={'count':'Count','mean':f'Mean {severity_ref}'})
                cluster_summary[f'Mean {severity_ref}'] = cluster_summary[f'Mean {severity_ref}'].round(3)
                st.write("Cluster summary (with severity labels):")
                st.dataframe(cluster_summary.sort_values(f'Mean {severity_ref}'))

                # Visual: cluster severity counts
                fig_sev = px.bar(cluster_summary, x='ClusterSeverity', y='Count', color='ClusterSeverity', title='Cluster counts by assigned severity', text='Count')
                st.plotly_chart(fig_sev, use_container_width=True)

# --- 4️⃣ Classification ---
with st.expander("🧠 Step 4: Random Forest Classification", expanded=False):
    col_options = [None] + cleaned_df.columns.tolist()
    # default to Cluster if present
    default_index = 0
    if 'Cluster' in cleaned_df.columns:
        default_index = col_options.index('Cluster')
    target_col = st.selectbox("Select target column (optional, defaults to Cluster)", col_options, index=default_index)

    if st.button("Train Classifier", key="classifier"):
        if target_col:
            target = cleaned_df[target_col]
            X = cleaned_df.drop(columns=[target_col])
        else:
            if 'Cluster' not in cleaned_df.columns:
                st.error("No 'Cluster' found. Run clustering or pick a target column.")
                st.stop()
            target = cleaned_df['Cluster']
            X = cleaned_df.drop(columns=['Cluster'])

        X_num = X.select_dtypes(include=[np.number])
        if X_num.shape[1] == 0:
            st.error("No numeric features available for classification.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_num, target, train_size=0.75, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)

            # Show accuracy clearly with colored indicator emoji
            st.success(f"✅ Classifier trained — Accuracy: {acc:.3f} {color_for_score(acc, higher_is_better=True)}")
            st.write({"Classification Accuracy": round(float(acc), 3)})

            # confusion matrix with named axes if possible
            cm = confusion_matrix(y_test, preds)
            labels = np.unique(np.concatenate([y_test.astype(str), preds.astype(str)]))
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Viridis', title='Confusion Matrix',
                               labels=dict(x="Predicted label", y="True label"))
            fig_cm.update_xaxes(tickvals=list(range(len(labels))), ticktext=labels.tolist())
            fig_cm.update_yaxes(tickvals=list(range(len(labels))), ticktext=labels.tolist())
            st.plotly_chart(fig_cm, use_container_width=True)

            # classification report as table
            try:
                report = classification_report(y_test, preds, output_dict=True)
                report_df = pd.DataFrame(report).T
                st.write("Classification report (precision / recall / f1 per label):")
                st.dataframe(report_df.round(3))
            except Exception:
                st.write("Classification report unavailable for these labels.")

            # feature importances
            importances = clf.feature_importances_
            feat_imp = pd.DataFrame({'Feature': X_num.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
            st.write("Top feature importances (Random Forest):")
            st.dataframe(feat_imp.head(20).reset_index(drop=True).round(4))
            fig_imp = px.bar(feat_imp.head(20), x='Feature', y='Importance', title='Top Feature Importances', text='Importance')
            st.plotly_chart(fig_imp, use_container_width=True)

            # Example predictions table (first 10) with feature values and predicted vs actual:
            test_examples = X_test.copy().reset_index(drop=True).iloc[:10]
            test_examples['Actual'] = y_test.reset_index(drop=True).iloc[:10]
            test_examples['Predicted'] = preds[:10]
            # if target was categorical and an encoder exists, try mapping back to original labels
            if target_col in encoders:
                le = encoders[target_col]
                try:
                    test_examples['Actual'] = le.inverse_transform(test_examples['Actual'].astype(int))
                    test_examples['Predicted'] = le.inverse_transform(test_examples['Predicted'].astype(int))
                except Exception:
                    pass
            st.write("Example predictions (first 10 rows):")
            st.dataframe(test_examples.round(4))

            # save classifier metrics
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

            st.success(f"✅ Validation complete — Mean Accuracy: {scores.mean():.3f} {color_for_score(scores.mean(), True)}")
            folds = list(range(1, len(scores) + 1))
            scores_df = pd.DataFrame({'Fold': folds, 'Accuracy': scores.round(4)})
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
                st.stop()
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

            st.success(f"✅ Regression complete — R²: {r2:.3f} {color_for_score(r2, True)}")
            st.write({"R²": round(float(r2), 3), "MAE": round(float(mae), 3), "RMSE": round(float(rmse), 3)})

            # Interpretability help
            with st.expander("What these regression metrics mean", expanded=False):
                st.write("- R²: fraction of variance explained by the model (closer to 1 is better).")
                st.write("- MAE: average absolute error between predicted and actual (lower is better).")
                st.write("- RMSE: root mean squared error, penalizes larger errors (lower is better).")

            # Actual vs Predicted scatter plot with a 45-degree reference line
            scatter_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
            fig = px.scatter(scatter_df, x='Actual', y='Predicted', trendline="ols",
                             title=f'Actual vs Predicted — target: {reg_target if reg_target else "Cluster"}',
                             labels={'Actual': f'Actual ({reg_target if reg_target else "Cluster"})', 'Predicted': 'Predicted'})
            st.plotly_chart(fig, use_container_width=True)

            # show few example predictions with errors
            examples = X_test.reset_index(drop=True).iloc[:10].copy()
            examples['Actual'] = y_test.reset_index(drop=True).iloc[:10]
            examples['Predicted'] = preds[:10]
            examples['Error'] = (examples['Predicted'] - examples['Actual']).round(4)
            st.write("Example regression predictions (first 10):")
            st.dataframe(examples.round(4))

            # feature importances
            importances = regr.feature_importances_
            feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
            st.write("Top regression feature importances:")
            st.dataframe(feat_imp.head(20).reset_index(drop=True).round(4))
            fig_imp = px.bar(feat_imp.head(20), x='Feature', y='Importance', title='Top Regression Feature Importances', text='Importance')
            st.plotly_chart(fig_imp, use_container_width=True)

            st.session_state['reg_r2'] = r2
            st.session_state['reg_metrics'] = {'r2': r2, 'mae': mae, 'rmse': rmse}

# --- 7️⃣ Final Summary ---
with st.expander("📜 Step 7: Final Summary", expanded=False):
    st.subheader("Performance Summary (named & color-coded)")
    metrics = {}
    if 'clf_acc' in st.session_state:
        metrics['Classification Accuracy'] = float(st.session_state['clf_acc'])
    if 'cv_scores' in st.session_state:
        metrics['Mean K-Fold Accuracy'] = float(st.session_state['cv_scores'].mean())
    if 'reg_r2' in st.session_state:
        metrics['Regression R²'] = float(st.session_state['reg_r2'])
    if 'reg_metrics' in st.session_state:
        metrics['Regression MAE'] = float(st.session_state['reg_metrics']['mae'])
        metrics['Regression RMSE'] = float(st.session_state['reg_metrics']['rmse'])

    if metrics:
        # display key KPIs as metric cards (left column)
        cols = st.columns(len(metrics))
        for (label, value), col in zip(metrics.items(), cols):
            # choose if higher is better for some metrics
            higher_better = not label.lower().startswith('regression mae') and not label.lower().startswith('regression rmse')
            emoji = color_for_score(value, higher_is_better=higher_better)
            col.metric(label=label, value=round(value, 3), delta=emoji)
        # also show a summary table
        summary_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        summary_df['Value'] = summary_df['Value'].apply(lambda x: round(float(x), 3))
        st.dataframe(summary_df)
        fig = px.bar(summary_df, x='Metric', y='Value', text='Value', color='Metric', title='Model Performance Summary (named metrics)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No computed metrics found. Run the modeling steps above to populate summary.")

# --- Export ---
with st.expander("💾 Export Cleaned Dataset", expanded=False):
    if st.button("Generate Download Link"):
        csv = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned CSV", csv, "cleaned_microplastic_dataset.csv", "text/csv")

st.markdown("---")
st.caption("Improvements: human-readable cluster severity labels, example prediction tables, clear KPI presentation and metric explanations to make results easy to understand.")
