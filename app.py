import joblib
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Phishing",
    page_icon="🛡️",
    layout="centered"
)

# Chargement du modèle mis en cache (évite de le recharger à chaque clic)
@st.cache_resource
def charger_modele():
    return joblib.load("modele_phishing_svc.joblib")

modele = charger_modele()

# En-tête de l'interface
st.title("🛡️ Scanner d'e-mails anti-phishing")
st.write("Collez le texte d'un e-mail ci-dessous pour analyser s'il est légitime ou suspect.")

# Zone de saisie utilisateur
email_texte = st.text_area(
    label="Contenu de l'e-mail :",
    height=220,
    placeholder="Collez ici l'objet et le corps de l'e-mail à vérifier..."
)

# Bouton de soumission
if st.button("Analyser l'e-mail", type="primary"):
    if not email_texte.strip():
        st.warning("Veuillez saisir ou coller un e-mail avant de lancer l'analyse.")
    else:
        # Le pipeline applique automatiquement TF-IDF puis LinearSVC
        prediction = modele.predict([email_texte])[0]
        score_marge = modele.decision_function([email_texte])[0]
        certitude = abs(score_marge)

        # Interprétation simple de la distance à la frontière de décision (0)
        if certitude >= 1.0:
            niveau_confiance = "Élevée"
        elif certitude >= 0.4:
            niveau_confiance = "Modérée"
        else:
            niveau_confiance = "Faible (cas limite)"

        st.divider()

        if prediction == 1:
            st.error("🚨 **Résultat : PHISHING DÉTECTÉ**")
        else:
            st.success("✅ **Résultat : E-MAIL LÉGITIME**")

        col1, col2 = st.columns(2)
        col1.metric("Score de décision (marge SVM)", f"{score_marge:.2f}")
        col2.metric("Confiance du modèle", niveau_confiance)

        st.caption(
            "Lecture du score : un score positif oriente vers le phishing, "
            "un score négatif vers un e-mail légitime. Plus le score est éloigné de 0, "
            "plus la classification est nette."
        )