import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Classification des Claims 📊",
    layout="wide"
)

# Fonction pour charger les données
@st.cache_data
def load_data():
    try:
        # Vérification de l'existence des fichiers avant de les charger
        data_dir = "data"
        required_files = ["train.csv", "test.csv", "X.csv", "y.csv", "train_data.csv"]
        
        # Vérifier si les fichiers requis existent
        for file in required_files:
            if not os.path.exists(os.path.join(data_dir, file)):
                st.error(f"Fichier manquant: {os.path.join(data_dir, file)}")
                return None
        
        # Charger les fichiers de base
        train = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test = pd.read_csv(os.path.join(data_dir, "test.csv"))
        X = pd.read_csv(os.path.join(data_dir, "X.csv"))
        y = pd.read_csv(os.path.join(data_dir, "y.csv"))
        train_data = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
        
        # Charger les fichiers optionnels
        data_dict = {
            "train": train,
            "test": test,
            "X": X,
            "y": y,
            "train_data": train_data,
        }
        
        # Charger les fichiers de prédiction s'ils existent
        optional_files = {
            "test": "test.csv",
            "summary_metrics": "summary_metrics.csv",
            "test_final_prediction": "test_final_prediction.csv",
            "test_pred_xgb": "test_pred_xgb.csv",
            "test_pred_rf": "test_pred_rf.csv",
            "test_pred_log": "test_pred_log.csv",
            "test_pred_svm": "test_pred_svm.csv",
            "test_pred_knn": "test_pred_knn.csv"
        }
        
        for key, filename in optional_files.items():
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                data_dict[key] = pd.read_csv(file_path)
            else:
                st.warning(f"Fichier optionnel non trouvé: {file_path}")
                data_dict[key] = None
        
        return data_dict
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        import traceback
        st.error(f"Détails: {traceback.format_exc()}")
        return None
# Fonction pour charger les modèles
@st.cache_resource
def load_models():
    try:
        models = {}
        model_files = {
            "XGBoost": "models/xgb_model.pkl",
            "Random Forest": "models/rf_model.pkl",
            "Logistic Regression": "models/log_model.pkl",
            "SVC": "models/svm_model.pkl",
            "KNN": "models/knn_model.pkl",
        }
        
        for name, file in model_files.items():
            if os.path.exists(file):
                with open(file, "rb") as f:
                    models[name] = pickle.load(f)
        
        return models
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        return {}

# Titre principal
st.title("🔍 Dashboard : Car Insurance Claim Prediction")
st.markdown("*Prédire si le titulaire déposera une réclamation dans les 6 prochains mois ou non.*")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Sélectionnez une page:",
    ["Aperçu des données", "Analyse exploratoire", "Performances des modèles", "Prédictions", "Analyse PCA","Prédiction personnalisée"]
)

# Chargement des données
data = load_data()
if data is None:
    st.warning("Impossible de charger les données. Veuillez vérifier les chemins des fichiers.")
    st.stop()

# Chargement des modèles (si disponibles)
models = load_models()

# Page: Aperçu des données
if page == "Aperçu des données":
    st.header("📋 Aperçu des Données")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informations sur les jeux de données")
        st.write(f"**Data initiale:** {data['train'].shape[0]} lignes, {data['train'].shape[1]} colonnes")
        st.write(f"**Data initiale après cleaning:** {data['train_data'].shape[0]} lignes, {data['train'].shape[1]} colonnes")
        st.write(f"**Data de test:** {data['test'].shape[0]} lignes, {data['test'].shape[1]} colonnes")
        
        
    
    with col2:
        st.subheader("Distribution de la classe cible")
        fig, ax = plt.subplots(figsize=(8, 6))
        class_counts = data['train']['is_claim'].value_counts()
        ax.pie(class_counts, labels=['Non-Claim', 'Claim'], autopct='%1.1f%%', 
               colors=['#66b3ff', '#ff9999'], explode=[0, 0.1])
        ax.set_title("Répartition des classes")
        st.pyplot(fig)
    # Types des colonnes
    if st.checkbox("Afficher les types de colonnes"):
        col_types = pd.DataFrame({
            'Type': data['train'].dtypes,
            'Non-Null Count': data['train'].count(),
            'Null Count': data['train'].isnull().sum()
        })
        st.dataframe(col_types)
    st.subheader("Aperçu des données d'entraînement")
    st.dataframe(data['train'].head(10))
    if st.checkbox("Afficher les statistiques descriptives"):
            st.subheader("Statistiques descriptives")
            st.dataframe(data['train'].describe())
    st.subheader("Aperçu des données aprés cleaning")
    st.dataframe(data['train_data'].head(10))
    if st.checkbox("Afficher les statistiques descriptives(après cleaning)"):
            st.subheader("Statistiques descriptives")
            st.dataframe(data['train_data'].describe())
   

# Page: Analyse exploratoire
elif page == "Analyse exploratoire":
    st.header("🔎 Analyse Exploratoire des Données")
    
    tab1, tab2, tab3 = st.tabs(["Distributions", "Corrélations", "Feature Importance"])
    
    with tab1:
        st.subheader("Distribution des variables numériques")
        st.subheader("Feature Engineering")
        
        # Afficher les booléens convertis
        boolean_cols = [col for col in data['train'].columns if 'is_' in col]
        
        if len(boolean_cols) > 0:
            st.write("Variables booléennes converties:")
            
            # Afficher la distribution des variables booléennes
            # Créer un DataFrame pour la vue d'ensemble
            bool_summary = pd.DataFrame(index=boolean_cols)
            
            for col in boolean_cols:
                if col in data['train'].columns:
                    bool_summary.loc[col, 'Pourcentage de Yes'] = (data['train'][col] == 'Yes').mean() * 100 if data['train'][col].dtype == 'object' else data['train'][col].mean() * 100
                    bool_summary.loc[col, 'Pourcentage de No'] = 100 - bool_summary.loc[col, 'Pourcentage de Yes']
            
            bool_summary = bool_summary.sort_values('Pourcentage de Yes', ascending=False)
            
            # Afficher avec Plotly
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=bool_summary.index,
                x=bool_summary['Pourcentage de Yes'],
                name='Yes',
                orientation='h',
                marker=dict(color='#ff9999')
            ))
            
            fig.add_trace(go.Bar(
                y=bool_summary.index,
                x=bool_summary['Pourcentage de No'],
                name='No',
                orientation='h',
                marker=dict(color='#66b3ff')
            ))
            
            fig.update_layout(
                title="Distribution des variables booléennes",
                xaxis_title="Pourcentage",
                barmode='stack',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        
        # Sélection de colonnes pour visualisation
        num_columns = data['X'].select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_cols = st.multiselect(
            "Sélectionnez les colonnes à visualiser:", 
            num_columns,
            default=num_columns[:5]
        )
        
        if selected_cols:
            col1, col2 = st.columns(2)
            with col1:
                # Histogrammes
                for col in selected_cols:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(data=data['train_data'], x=col, hue='is_claim', kde=True, ax=ax)
                    ax.set_title(f"Distribution de {col}")
                    st.pyplot(fig)
            
            with col2:
                # Boxplots
                for col in selected_cols:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.boxplot(data=data['train_data'], x='is_claim', y=col, ax=ax)
                    ax.set_title(f"Boxplot de {col} par classe")
                    ax.set_xticklabels(['Non-Claim', 'Claim'])
                    st.pyplot(fig)
    
    with tab2:
        st.subheader("Matrice de Corrélation")
        
        # Matrice de corrélation
        corr_matrix = data['X'].corr()
        subset=corr_matrix.abs().mean().sort_values(ascending=False).index
        corr_matrix = corr_matrix.loc[subset, subset]
        
        fig, ax = plt.subplots(figsize=(18, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(data=corr_matrix, annot=False, fmt=".2f", cmap='coolwarm', square=True,linewidths=1)
        
        ax.set_title('Matrice de Corrélation des Variables')
        st.pyplot(fig)
        
        # Top corrélations
        st.subheader("Bar Corrélations")
        
        def get_top_correlations(corr_matrix):
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], 
                         corr_matrix.iloc[i, j])
                    )
            return corr_pairs
        
        top_corrs = get_top_correlations(corr_matrix)

        top_corrs_df = pd.DataFrame(
            top_corrs, columns=['Variable 1', 'Variable 2', 'Corrélation']
        ).sort_values(by='Corrélation', ascending=False)
        
        fig = px.bar(
            top_corrs_df, 
            x='Corrélation', 
            y=[f"{row['Variable 1']} - {row['Variable 2']}" for _, row in top_corrs_df.iterrows()],
            orientation='h',
            color='Corrélation',
            color_continuous_scale=px.colors.diverging.RdBu_r,  # <--- le coupable
            range_color=[-1, 1]
        )
        fig.update_layout(height=600, title_text=" corrélations")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_corrs_df)

    
    with tab3:
        st.subheader("Importance des Features")
        
        # Recréer le SelectKBest pour l'importance des features
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_classif
        
        X_arr = data['X'].values
        y_arr = data['y'].values.ravel()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
        
        selector = SelectKBest(score_func=f_classif, k=20)
        selector.fit(X_scaled, y_arr)
        
        # Créer un DataFrame pour l'importance
        feature_scores = pd.DataFrame({
            'Feature': data['X'].columns,
            'Score': selector.scores_
        })
        feature_scores = feature_scores.sort_values('Score', ascending=False).head(20)
        
        # Afficher avec Plotly
        fig = px.bar(
            feature_scores, 
            x='Score', 
            y='Feature',
            orientation='h',
            color='Score',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600, title_text="Top 20 des features importantes (ANOVA F-value)")
        st.plotly_chart(fig, use_container_width=True)

# Page: Performances des modèles
elif page == "Performances des modèles":
    st.header("📈 Performances des Modèles")
    
    # Metrics globales
    st.subheader("Comparaison des métriques")
    
    # Visualisation des métriques
    metrics = data['summary_metrics'].set_index('Model')
    
    # Graphique de comparaison
    fig = go.Figure()
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for model in metrics.index:
        fig.add_trace(go.Bar(
            x=metrics_cols,
            y=metrics.loc[model, metrics_cols],
            name=model
        ))
    
    fig.update_layout(
        title="Comparaison des métriques par modèle",
        xaxis_title="Métrique",
        yaxis_title="Score",
        legend_title="Modèle",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Visualisation sous forme de tableau
    st.dataframe(metrics)
    
    # Matrices de confusion
    st.subheader("Matrices de confusion")
    
    col1, col2, col3 = st.columns(3)
    
    # Fonction pour créer les matrices de confusion à partir des prédictions
    def create_confusion_matrix(true_labels, predictions, title):
        cm = confusion_matrix(true_labels, predictions)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        return fig
    
    # Si les prédictions sur le jeu de test étaient disponibles
    if not len(models) > 0:
        # Utiliser le jeu de validation pour les matrices de confusion
        # Note: Cette partie nécessiterait d'avoir sauvegardé les prédictions de validation
        st.info("Les matrices de confusion nécessitent d'avoir sauvegardé les prédictions de validation.")
    else:
        # Utiliser les images sauvegardées si disponibles
        try:
            col1.image("visuals/confusion_xgboost.png", caption="Matrice de confusion - XGBoost")
            col2.image("visuals/confusion_rf.png", caption="Matrice de confusion - Random Forest")
            col3.image("visuals/confusion_log.png", caption="Matrice de confusion - Logistic Regression")
            
            col1, col2, col3 = st.columns(3)
            col1.image("visuals/confusion_svm.png", caption="Matrice de confusion - SVM")
            col2.image("visuals/confusion_knn.png", caption="Matrice de confusion - KNN")
            col3.image("visuals/confusion_mlp.png", caption="Matrice de confusion - MLP")
        except:
            st.warning("Les images des matrices de confusion ne sont pas disponibles.")

# Page: Prédictions
elif page == "Prédictions":
    st.header("🔮 Analyse des Prédictions")
    
    # Analyse des prédictions finales
    st.subheader("Prédictions sur la data de test lorsque au moins un modèle prédit 'claim'")
    
    # Tableau de prédictions
    prediction_cols = [col for col in data['test_final_prediction'].columns if 'is_claim' in col]
    
    # Créer une colonne de vote majoritaire
    if len(prediction_cols) > 0:
        test_predictions = data['test_final_prediction'][prediction_cols].copy()
        test_predictions['vote_majoritaire'] = test_predictions.mean(axis=1).round()
        
        # Afficher les prédictions
        st.dataframe(test_predictions[(test_predictions['is_claim_xgb'] == 1) | (test_predictions['is_claim_rf'] == 1) 
| (test_predictions['is_claim_log'] == 1)| (test_predictions['is_claim_svm'] == 1) 
| (test_predictions['is_claim_knn'] == 1)])
        
        # Distribution des prédictions
        st.subheader("Distribution des prédictions")
        
        # Compter les prédictions positives par modèle
        model_counts = test_predictions.sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=model_counts.index,
            y=model_counts.values,
            labels={'x': 'Modèle', 'y': 'Nombre de prédictions "claim"'},
            title="Nombre de prédictions positives par modèle"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Histogramme des votes
        fig = px.histogram(
            test_predictions.sum(axis=1), 
            title="Distribution du nombre de modèles prédisant 'claim'",
            labels={'value': 'Nombre de modèles prédisant "claim"', 'count': 'Nombre d\'instances'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Sous forme de tableau")
        st.dataframe(test_predictions.sum(axis=1).value_counts().reset_index(name='count').rename(columns={'index': 'Nombre de modèles prédisant "claim"'}))
        
        # Prédictions unanimes
        st.subheader("Prédictions unanimes")
        unanimous_positive = test_predictions[test_predictions.sum(axis=1) == len(prediction_cols) - 1]  
        unanimous_negative = test_predictions[test_predictions.sum(axis=1) == 0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prédictions unanimes 'claim' (80%)", len(unanimous_positive))
        with col2:
            st.metric("Prédictions unanimes 'non-claim'", len(unanimous_negative))
        
        # Afficher les instances avec prédictions unanimes positives
        if len(unanimous_positive) > 0:
            st.subheader("Instances unanimement prédites comme 'claim'")
            unanimously_positive_indices = unanimous_positive.index
            st.dataframe(data['test'].loc[unanimously_positive_indices])
    else:
        st.warning("Aucune colonne de prédiction trouvée dans les données.")

# Page: Analyse PCA
elif page == "Analyse PCA":
    st.header("🧪 Analyse PCA")
    st.subheader("Analyse en Composantes Principales (PCA)")
        
        # Recréer la PCA
    from sklearn.preprocessing import StandardScaler
        
    X_arr = data['X'].values
    y_arr = data['y'].values.ravel()
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
        
    # Appliquer PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
        
    # Variance expliquée
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
    fig = px.line(
            x=range(1, len(cumulative_variance)+1),
            y=cumulative_variance,
            markers=True,
            title="Variance expliquée cumulée par composante PCA",
            labels={"x": "Nombre de composantes", "y": "Variance expliquée cumulée"}
        )
        
    fig.add_hline(y=0.95, line_dash="dash", line_color="red", 
                     annotation_text="95% de variance expliquée", 
                     annotation_position="bottom right")
        
    st.plotly_chart(fig, use_container_width=True)
        
    # Visualisation 2D des données après PCA
    if len(X_pca) > 0:
            st.subheader("Visualisation des données après PCA")
            
            # Créer un DataFrame pour la visualisation
            pca_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Classe': y_arr
            })
            
            fig = px.scatter(
                pca_df, 
                x='PC1', 
                y='PC2', 
                color='Classe',
                color_discrete_map={0: '#66b3ff', 1: '#ff9999'},
                title="Projection des données sur les deux premières composantes principales",
                labels={'Classe': 'Classe (0: Non-Claim, 1: Claim)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Loadings des composantes principales
            st.subheader("Contributions des features aux composantes principales")
            
            n_features = min(10, len(data['X'].columns))
            loadings = pca.components_[:2, :].T  # Prendre les 2 premières composantes
            
            loadings_df = pd.DataFrame({
                'Feature': data['X'].columns,
                'PC1': loadings[:, 0],
                'PC2': loadings[:, 1]
            })
            
            # Tri par importance sur PC1
            loadings_df = loadings_df.sort_values('PC1', key=abs, ascending=False).head(n_features)
            
            fig = px.bar(
                loadings_df,
                x='Feature',
                y=['PC1', 'PC2'],
                barmode='group',
                title=f"Contributions des {n_features} features les plus importantes aux PC1 et PC2",
                labels={'value': 'Contribution', 'variable': 'Composante principale'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
# Page: Prédiction personnalisée
elif page == "Prédiction personnalisée":
    st.header("🔮 Prédiction Personnalisée")
    
    # Formulaire pour la saisie des données
    st.subheader("Entrez les valeurs des features")
    
    # Créer un formulaire pour la saisie des données
    with st.form(key='prediction_form'):
        input_data = {}
        for col in data['X'].columns:
            input_data[col] = st.number_input(f"Valeur de {col}", value=0.0, step=0.01)
        
        submit_button = st.form_submit_button(label='Faire une prédiction')
    
    if submit_button:
        # Convertir les données d'entrée en DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Normaliser les données d'entrée
        
        
        # Faire des prédictions avec chaque modèle
        predictions = {}
        for model_name, model in models.items():
            predictions[model_name] = model.predict(input_df)[0]
        
        # Afficher les résultats
        st.subheader("Résultats de la prédiction")
        for model_name, prediction in predictions.items():
            st.write(f"{model_name}: {'Claim' if prediction == 1 else 'Non-Claim'}")
        st.subheader("Vote majoritaire")
        vote_majoritaire = sum(predictions.values()) / len(predictions)
        st.write(f"Vote majoritaire: {'Claim' if vote_majoritaire >= 0.5 else 'Non-Claim'}")

    
    
        
            

# Pied de page
st.markdown("---")
st.markdown("*Dashboard créé avec Streamlit pour l'analyse des données de classification*")