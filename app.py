import html
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Phishing Explicable",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def charger_modele():
    return joblib.load("modele_phishing_svc.joblib")


def extraire_explications_svm(pipeline, texte):
    """Calcule la contribution exacte de chaque mot/bigramme dans la décision du LinearSVC."""
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    vecteur = vectorizer.transform([texte])
    score_total = float(pipeline.decision_function([texte])[0])
    prediction = int(pipeline.predict([texte])[0])
    biais = float(clf.intercept_[0])

    noms_features = vectorizer.get_feature_names_out()
    indices = vecteur.nonzero()[1]
    valeurs_tfidf = vecteur.data
    poids_svm = clf.coef_[0][indices]
    contributions = valeurs_tfidf * poids_svm

    df_contrib = pd.DataFrame({
        "Terme": noms_features[indices],
        "Score TF-IDF": valeurs_tfidf,
        "Poids SVM": poids_svm,
        "Contribution": contributions
    }).sort_values(by="Contribution", ascending=False).reset_index(drop=True)

    return prediction, score_total, biais, df_contrib


def analyser_signaux_contextuels(texte, nb_termes_reconnus):
    """Signaux simples de contexte (informatifs, indépendants du modèle SVM)."""
    mots = re.findall(r"[A-Za-zÀ-ÿ0-9]+", texte)
    nb_mots = len(mots)
    urls = re.findall(r"(?:https?://|www\.)\S+", texte, flags=re.IGNORECASE)
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", texte)
    lettres = [c for c in texte if c.isalpha()]
    ratio_maj = (sum(c.isupper() for c in lettres) / len(lettres) * 100) if lettres else 0.0
    nb_exclamations = texte.count("!")

    return {
        "nb_mots": nb_mots,
        "nb_termes_reconnus": nb_termes_reconnus,
        "nb_urls": len(urls),
        "urls": urls,
        "nb_ips": len(ips),
        "ratio_maj": ratio_maj,
        "nb_exclamations": nb_exclamations
    }


def surligner_email(texte, df_contrib):
    """Génère un rendu HTML de l'e-mail avec surlignage selon la contribution de chaque mot."""
    if df_contrib.empty:
        return html.escape(texte).replace("\n", "<br>")

    dico_contrib = dict(zip(df_contrib["Terme"], df_contrib["Contribution"]))
    morceaux = re.findall(r"[A-Za-zÀ-ÿ0-9]+|[^A-Za-zÀ-ÿ0-9]+", texte)
    scores_pos = [0.0] * len(morceaux)

    # Repérage des positions des mots
    idx_mots = [i for i, m in enumerate(morceaux) if re.fullmatch(r"[A-Za-zÀ-ÿ0-9]+", m)]

    # 1. Ajout des contributions des unigrammes (mots seuls)
    for pos in idx_mots:
        mot = morceaux[pos].lower()
        if mot in dico_contrib:
            scores_pos[pos] += dico_contrib[mot]

    # 2. Répartition des contributions des bigrammes consécutifs sur leurs 2 mots
    for k in range(len(idx_mots) - 1):
        p1, p2 = idx_mots[k], idx_mots[k + 1]
        bigramme = f"{morceaux[p1].lower()} {morceaux[p2].lower()}"
        if bigramme in dico_contrib:
            scores_pos[p1] += dico_contrib[bigramme] / 2.0
            scores_pos[p2] += dico_contrib[bigramme] / 2.0

    max_abs = max((abs(s) for s in scores_pos), default=0.0)
    if max_abs == 0:
        return html.escape(texte).replace("\n", "<br>")

    blocs_html = []
    for morceau, score in zip(morceaux, scores_pos):
        texte_echappe = html.escape(morceau).replace("\n", "<br>")
        if abs(score) > 0.01:
            # Intensité de couleur proportionnelle à l'impact du mot (entre 0.15 et 0.65)
            alpha = 0.15 + 0.50 * (abs(score) / max_abs)
            couleur = f"rgba(220, 53, 69, {alpha:.2f})" if score > 0 else f"rgba(25, 135, 84, {alpha:.2f})"
            direction = "Phishing" if score > 0 else "Légitime"
            blocs_html.append(
                f'<span title="{direction} (contribution : {score:+.3f})" '
                f'style="background-color:{couleur}; padding:1px 3px; border-radius:4px;">'
                f'{texte_echappe}</span>'
            )
        else:
            blocs_html.append(texte_echappe)

    return "".join(blocs_html)


def tracer_graphique_contributions(df_contrib, top_n=8):
    """Trace un graphique en barres divergentes des termes les plus influents."""
    top_phishing = df_contrib[df_contrib["Contribution"] > 0].head(top_n)
    top_legit = df_contrib[df_contrib["Contribution"] < 0].tail(top_n)
    df_visu = pd.concat([top_phishing, top_legit]).sort_values(by="Contribution", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(df_visu) * 0.35)))
    couleurs = ["#d9534f" if v > 0 else "#2e8b57" for v in df_visu["Contribution"]]

    ax.barh(df_visu["Terme"], df_visu["Contribution"], color=couleurs)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_title("Termes les plus influents dans la décision du modèle")
    ax.set_xlabel("Contribution au score (← Légitime | Phishing →)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


# --- INTERFACE UTILISATEUR ---
modele = charger_modele()

st.title("🛡️ Scanner d'e-mails anti-phishing (avec explicabilité)")
st.write(
    "Collez un e-mail ci-dessous pour obtenir la prédiction du modèle **LinearSVC** "
    "ainsi que le détail des éléments qui justifient sa décision."
)

# Exemples prêts à l'emploi pour faciliter les tests
col_ex1, col_ex2 = st.columns(2)
if col_ex1.button("Charger un exemple suspect (Phishing)"):
    st.session_state["email_input"] = (
        "URGENT: Your bank account has been suspended due to unusual activity! "
        "Please click the link below immediately to verify your password and billing details: "
        "http://secure-verify-account-login.example.com"
    )
if col_ex2.button("Charger un exemple normal (Légitime)"):
    st.session_state["email_input"] = (
        "Hi team, thanks for sending the meeting notes from Tuesday. "
        "I reviewed the schedule and we can finalize the project report tomorrow afternoon. "
        "Let me know if you need anything else before then. Best regards."
    )

email_texte = st.text_area(
    label="Contenu de l'e-mail à analyser :",
    key="email_input",
    height=180,
    placeholder="Collez ici l'objet et le corps de l'e-mail..."
)

if st.button("Analyser l'e-mail", type="primary"):
    if not email_texte.strip():
        st.warning("Veuillez saisir un texte avant de lancer l'analyse.")
    else:
        prediction, score_total, biais, df_contrib = extraire_explications_svm(modele, email_texte)
        signaux = analyser_signaux_contextuels(email_texte, len(df_contrib))

        certitude = abs(score_total)
        if certitude >= 1.0:
            niveau_confiance = "Élevée"
        elif certitude >= 0.4:
            niveau_confiance = "Modérée"
        else:
            niveau_confiance = "Faible (cas limite)"

        somme_phishing = df_contrib.loc[df_contrib["Contribution"] > 0, "Contribution"].sum()
        somme_legit = df_contrib.loc[df_contrib["Contribution"] < 0, "Contribution"].sum()

        st.divider()

        # 1. VERDICT PRINCIPAL
        if prediction == 1:
            st.error("🚨 **Verdict : PHISHING DÉTECTÉ**")
        else:
            st.success("✅ **Verdict : E-MAIL LÉGITIME**")

        if df_contrib.empty:
            st.warning(
                "Aucun mot de cet e-mail ne figure dans le vocabulaire appris par le modèle "
                "(ex. langue différente ou texte trop court). La décision repose uniquement sur le biais par défaut."
            )

        # 2. DÉCOMPOSITION MATHÉMATIQUE DU SCORE
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score final SVM", f"{score_total:+.2f}")
        c2.metric("Confiance", niveau_confiance)
        c3.metric("Poids « Phishing »", f"{somme_phishing:+.2f}")
        c4.metric("Poids « Légitime »", f"{somme_legit:+.2f}")

        st.caption(
            f"Décomposition exacte : Biais initial ({biais:+.2f}) "
            f"+ Termes suspects ({somme_phishing:+.2f}) "
            f"+ Termes rassurants ({somme_legit:+.2f}) = **{score_total:+.2f}** "
            f"(seuil de bascule à 0.00)."
        )

        # 3. SURLIGNAGE DU TEXTE ET GRAPHIQUE CÔTE À CÔTE
        st.subheader("1. Pourquoi le modèle a-t-il pris cette décision ?")
        col_gauche, col_droite = st.columns([1.1, 1])

        with col_gauche:
            st.markdown("**Lecture surlignée de l'e-mail** *(survolez un mot pour voir son impact)*")
            st.markdown(
                "<small>🟥 Rouge : tire vers Phishing &nbsp;&nbsp;|&nbsp;&nbsp; 🟩 Vert : tire vers Légitime</small>",
                unsafe_allow_html=True
            )
            html_surligne = surligner_email(email_texte, df_contrib)
            st.markdown(
                f'<div style="border:1px solid #ddd; padding:14px; border-radius:8px; '
                f'line-height:1.7; background-color:rgba(128,128,128,0.05);">{html_surligne}</div>',
                unsafe_allow_html=True
            )

        with col_droite:
            st.markdown("**Top des mots et expressions (bigrammes) décisifs**")
            if not df_contrib.empty:
                fig = tracer_graphique_contributions(df_contrib, top_n=8)
                st.pyplot(fig)
                plt.close(fig)

        # 4. INDICATEURS DE CONTEXTE (COMPLEMENTAIRES AU MODELE)
        st.subheader("2. Indicateurs complémentaires de vigilance")
        st.caption(
            "Ces indicateurs fournissent un contexte utile à l'utilisateur, mais ne sont pas "
            "utilisés directement comme variables d'entrée par le modèle TF-IDF + LinearSVC."
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Longueur", f"{signaux['nb_mots']} mots")
        k2.metric("Termes reconnus", signaux["nb_termes_reconnus"])
        k3.metric("Liens (URLs) détectés", signaux["nb_urls"])
        k4.metric("Majuscules / « ! »", f"{signaux['ratio_maj']:.1f}% / {signaux['nb_exclamations']}")

        alertes = []
        if signaux["nb_urls"] > 0:
            alertes.append(f"Présence de **{signaux['nb_urls']} lien(s)** dans le message : vérifiez toujours le nom de domaine avant de cliquer.")
        if signaux["nb_ips"] > 0:
            alertes.append("Présence d'une **adresse IP directe** dans le texte (pratique fréquente en phishing).")
        if signaux["ratio_maj"] > 20:
            alertes.append(f"Proportion élevée de **majuscules ({signaux['ratio_maj']:.1f} %)**, souvent associée à un ton d'urgence artificielle.")
        if signaux["nb_mots"] > 0 and signaux["nb_termes_reconnus"] < 3:
            alertes.append("Très peu de mots ont été reconnus par le modèle : vérifiez que l'e-mail est bien dans la même langue que le jeu d'entraînement.")

        for alerte in alertes:
            st.info(f"ℹ️ {alerte}")

        # 5. TABLEAU TECHNIQUE COMPLET
        with st.expander("Voir le tableau complet des contributions (mode technique)"):
            st.write(
                "Chaque ligne correspond à un mot ou bigramme présent dans l'e-mail et connu du vocabulaire TF-IDF. "
                "**Contribution = Score TF-IDF × Poids SVM**."
            )
            st.dataframe(
                df_contrib.style.format({
                    "Score TF-IDF": "{:.4f}",
                    "Poids SVM": "{:+.4f}",
                    "Contribution": "{:+.4f}"
                }),
                width=True
            )