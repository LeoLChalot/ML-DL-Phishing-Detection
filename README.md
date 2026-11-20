# Phishing Scanner

## Éxplicabilité

LinearSVC (combiné à TF-IDF) est mathématiquement transparent. 

Sa décision finale repose sur une simple somme pondérée :

$$Score = b + \sum_{i}(WeigthModel_{i} * score_tfidf_{i})$$

Chaque mot ou bigramme reconnu dans l'e-mail apporte donc une contribution exacte :
- **Contribution positive ($> 0$) :** le terme pousse la prédiction vers Phishing.
- **Contribution négative ($< 0$) :** le terme rassure le modèle et pousse vers Légitime.
- **$b$ (biais / intercept) :** point de départ neutre du modèle avant même de lire les mots.