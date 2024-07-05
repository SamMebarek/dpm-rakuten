import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

taux_inflation_mensuels = {
    "Jan 2020": 1.49,
    "Feb 2020": 1.43,
    "Mar 2020": 0.67,
    "Apr 2020": 0.33,
    "May 2020": 0.36,
    "Jun 2020": 0.20,
    "Jul 2020": 0.78,
    "Aug 2020": 0.22,
    "Sep 2020": 0.05,
    "Oct 2020": 0.05,
    "Nov 2020": 0.20,
    "Dec 2020": 0.02,
    "Jan 2021": 0.60,
    "Feb 2021": 0.80,
    "Mar 2021": 1.10,
    "Apr 2021": 1.20,
    "May 2021": 1.40,
    "Jun 2021": 1.50,
    "Jul 2021": 1.20,
    "Aug 2021": 1.30,
    "Sep 2021": 1.50,
    "Oct 2021": 1.70,
    "Nov 2021": 2.00,
    "Dec 2021": 1.60,
    "Jan 2022": 2.90,
    "Feb 2022": 3.60,
    "Mar 2022": 4.50,
    "Apr 2022": 4.80,
    "May 2022": 5.20,
    "Jun 2022": 5.80,
    "Jul 2022": 6.10,
    "Aug 2022": 5.90,
    "Sep 2022": 5.60,
    "Oct 2022": 6.20,
    "Nov 2022": 6.30,
    "Dec 2022": 5.90,
    "Jan 2023": 5.99,
    "Feb 2023": 6.28,
    "Mar 2023": 5.70,
    "Apr 2023": 5.88,
    "May 2023": 5.12,
    "Jun 2023": 4.53,
    "Jul 2023": 4.29,
    "Aug 2023": 4.86,
    "Sep 2023": 4.90,
    "Oct 2023": 3.98,
    "Nov 2023": 3.47,
    "Dec 2023": 3.71,
    "Jan 2024": 3.13,
    "Feb 2024": 2.96,
    "Mar 2024": 2.29,
    "Apr 2024": 2.19,
    "May 2024": 2.27,
}


class Produit:
    """
    Classe de base pour les produits.
    """

    def __init__(
        self,
        plage_prix_initiale,
        elasticite,
        categorie,
        sku,
        date_lancement,
        nombre_simulations=10000,
    ):
        self.prix_initial = np.random.uniform(*plage_prix_initiale)
        self.elasticite = elasticite
        self.categorie = categorie
        self.sku = sku
        self.date_lancement = date_lancement
        self.nombre_simulations = nombre_simulations

    def appliquer_inflation(self, prix, date):
        """
        Applique l'inflation au prix du produit en utilisant les taux d'inflation mensuels.

        Args:
            prix (float): Le prix actuel du produit.
            date (datetime): La date actuelle pour obtenir le taux d'inflation correct.

        Returns:
            float: Le prix après application de l'inflation.
        """
        date_str = date.strftime(
            "%b %Y"
        )  # Format de la date pour correspondre aux clés du dictionnaire
        inflation_rate = (
            taux_inflation_mensuels.get(date_str, 0) / 100
        )  # Taux d'inflation mensuel
        return prix * (1 + inflation_rate), inflation_rate

    def appliquer_marche_aleatoire(self, prix):
        """
        Applique une marche aleatoire au prix du produit.

        Args:
            prix (float): Le prix actuel du produit.

        Returns:
            float: Le prix moyen après application de la marche aleatoire.
        """
        chemins_prix = []
        for _ in range(self.nombre_simulations):
            choc = np.random.normal(0, 0.05)  # moyenne = 0, ecart-type = 5%
            chemins_prix.append(prix * np.exp(choc))
        return np.mean(chemins_prix)

    def simuler_changement_demande(self, prix):
        """
        Simule un changement de la demande et ajuste le prix en consequence.

        Args:
            prix (float): Le prix actuel du produit.

        Returns:
            float: Le prix après ajustement en fonction de l'elasticite.
        """
        changement_demande = np.random.uniform(
            -0.1, 0.1
        )  # Changement aleatoire de la demande entre -10% et +10%
        changement_prix = changement_demande / self.elasticite
        return prix * (1 + changement_prix)

    def generer_serie_temporelle(self, nombre_observations, nombre_total_mois):
        """
        Genere une serie temporelle de prix pour le produit.

        Args:
            nombre_observations (int): Le nombre d'observations pour generer la serie temporelle.
            nombre_total_mois (int): Le nombre total de mois pour generer les series temporelles.

        Returns:
            list: Une liste des prix pour chaque observation.
        """
        prix = []
        taux_inflation_liste = []
        for t in range(nombre_observations):
            date_vente = self.date_lancement + timedelta(
                days=(t * (nombre_total_mois * 30) // nombre_observations)
            )
            prix_actuel = self.prix_initial
            prix_actuel, inflation_rate = self.appliquer_inflation(
                prix_actuel, date_vente
            )
            prix_actuel = self.appliquer_depreciation(prix_actuel, t / 12)
            prix_actuel = self.appliquer_saisonnalite(prix_actuel, t / 12)
            prix_actuel = self.simuler_changement_demande(prix_actuel)
            prix_actuel = self.appliquer_marche_aleatoire(prix_actuel)
            prix.append(prix_actuel)
            taux_inflation_liste.append(inflation_rate)
        return prix, taux_inflation_liste

    def calculer_age_produit(self, date_reference):
        """
        Calcule l'age du produit en jours depuis la date de lancement jusqu'à une date de reference.

        Args:
            date_reference (datetime): La date de reference pour le calcul de l'age.

        Returns:
            int: L'age du produit en jours.
        """
        return (date_reference - self.date_lancement).days

    def determiner_categorie_age_produit(self, age_jours):
        """
        Determine la categorie d'age du produit en fonction de l'age en jours.

        Args:
            age_jours (int): L'age du produit en jours.

        Returns:
            str: La categorie d'age du produit ('Nouveau', 'Moyen', 'Ancien').
        """
        if self.categorie == "Electronique":
            if age_jours < 180:
                return "Nouveau"
            elif 180 <= age_jours < 548:
                return "Moyen"
            else:
                return "Ancien"
        elif self.categorie == "Livres":
            if age_jours < 90:
                return "Nouveau"
            elif 90 <= age_jours < 180:
                return "Moyen"
            else:
                return "Ancien"
        elif self.categorie == "Vetements":
            if age_jours < 90:
                return "Nouveau"
            elif 90 <= age_jours < 180:
                return "Moyen"
            else:
                return "Ancien"
        return "Inconnu"

    def generer_temps_en_rayon(self, categorie_age_produit):
        """
        Genere la durée de temps en rayon pour le produit en fonction de sa catégorie d'âge.

        Args:
            categorie_age_produit (str): La catégorie d'âge du produit ('Nouveau', 'Moyen', 'Ancien').

        Returns:
            float: Le nombre de jours que le produit reste en rayon.
        """
        if self.categorie == "Electronique":
            if categorie_age_produit == "Nouveau":
                return np.random.weibull(1.5) * 30  # Moins de 30 jours
            elif categorie_age_produit == "Moyen":
                return np.random.weibull(1.2) * 180  # Moins de 6 mois
            else:
                return np.random.weibull(1.0) * 365  # Moins d'un an
        elif self.categorie == "Livres":
            if categorie_age_produit == "Nouveau":
                return np.random.normal(45, 15)  # Environ 1.5 mois
            elif categorie_age_produit == "Moyen":
                return np.random.normal(120, 30)  # Environ 4 mois
            else:
                return np.random.normal(240, 60)  # Environ 8 mois
        elif self.categorie == "Vetements":
            if categorie_age_produit == "Nouveau":
                return np.random.weibull(1.5) * 90  # Moins de 3 mois
            elif categorie_age_produit == "Moyen":
                return np.random.weibull(1.2) * 180  # Moins de 6 mois
            else:
                return np.random.weibull(1.0) * 270  # Moins de 9 mois
        return 0


class Electronique(Produit):
    def appliquer_depreciation(self, prix, t):
        facteur_rarete = np.random.uniform(0.9, 1.1)
        if t > 0.5:
            return prix * np.exp(-0.2 * (t - 0.5)) * facteur_rarete
        return prix

    def appliquer_saisonnalite(self, prix, t):
        facteur_demande = 1 + 0.1 * np.sin(2 * np.pi * t)
        return prix * facteur_demande


class Vetements(Produit):
    def appliquer_depreciation(self, prix, t):
        facteur_rarete = np.random.uniform(0.9, 1.1)
        return max(
            0, prix * (1 - 0.1 * t + 0.3 * np.sin(2 * np.pi * t / 3)) * facteur_rarete
        )

    def appliquer_saisonnalite(self, prix, t):
        facteur_demande = 1 + 0.15 * np.sin(2 * np.pi * t)
        return prix * facteur_demande


class Livres(Produit):
    def appliquer_depreciation(self, prix, t):
        facteur_rarete = np.random.uniform(0.95, 1.05)
        return prix * facteur_rarete

    def appliquer_saisonnalite(self, prix, t):
        facteur_demande = 1 + 0.05 * np.sin(2 * np.pi * t)
        return prix * facteur_demande


def generer_donnees(
    nombre_observations, nombre_skus_par_categorie, date_debut, nombre_total_mois
):
    """
    Genere des donnees synthetiques pour plusieurs categories de produits.

    Args:
        nombre_observations (int): Le nombre total d'observations.
        nombre_skus_par_categorie (int): Le nombre de SKUs par categorie.
        date_debut (datetime): La date de debut pour les series temporelles.
        nombre_total_mois (int): Le nombre total de mois pour generer les series temporelles.

    Returns:
        pd.DataFrame: Un DataFrame contenant les donnees generees.
    """
    plages_prix_initiales = {
        "Electronique": (200, 1000),
        "Vetements": (20, 200),
        "Livres": (5, 50),
    }
    elasticites = {
        "Electronique": 1.5,  # Elastique
        "Vetements": 0.5,  # Inelastique
        "Livres": 1.0,  # Elasticite unitaire
    }
    categories = {
        "Electronique": Electronique,
        "Vetements": Vetements,
        "Livres": Livres,
    }

    donnees = []
    nombre_total_skus = nombre_skus_par_categorie * len(categories)
    observations_par_sku = nombre_observations // nombre_total_skus

    dates_lancement = []

    # Generer les dates de lancement pour les premiers 70% des SKUs
    for _ in range(int(0.7 * nombre_total_skus)):
        dates_lancement.append(date_debut + timedelta(days=np.random.normal(0, 30)))

    # Generer les dates de lancement pour les derniers 30% des SKUs
    for _ in range(int(0.3 * nombre_total_skus)):
        dates_lancement.append(
            date_debut
            + timedelta(days=np.random.normal(nombre_total_mois * 30 / 3, 30))
        )

    # S'assurer d'avoir exactement `nombre_total_skus` dates de lancement
    while len(dates_lancement) < nombre_total_skus:
        dates_lancement.append(
            date_debut
            + timedelta(days=np.random.normal(nombre_total_mois * 30 / 3, 30))
        )

    np.random.shuffle(dates_lancement)

    compteur_sku = 0
    for nom_categorie, ClasseProduit in categories.items():
        for sku in range(nombre_skus_par_categorie):
            date_lancement = dates_lancement[compteur_sku]
            compteur_sku += 1
            produit = ClasseProduit(
                plages_prix_initiales[nom_categorie],
                elasticites[nom_categorie],
                nom_categorie,
                f"{nom_categorie}_SKU_{sku}",
                date_lancement,
            )
            prix, taux_inflation_liste = produit.generer_serie_temporelle(
                observations_par_sku, nombre_total_mois
            )

            for i, prix_mois in enumerate(prix):
                date_vente = date_lancement + timedelta(
                    days=(i * (nombre_total_mois * 30) // observations_par_sku)
                )
                age_produit_en_jours = produit.calculer_age_produit(date_vente)
                categorie_age_produit = produit.determiner_categorie_age_produit(
                    age_produit_en_jours
                )
                temps_depuis_lancement = age_produit_en_jours

                # Simuler NombreDeJoursEnRayon en fonction de la categorie_age_produit
                nombre_de_jours_en_rayon = produit.generer_temps_en_rayon(
                    categorie_age_produit
                )

                # Arreter la simulation si la date de vente depasse la date de fin
                if date_vente > date_debut + timedelta(days=nombre_total_mois * 30):
                    break

                annee = date_vente.year
                mois = date_vente.month
                saison = (
                    mois % 12
                ) // 3 + 1  # Categorisation approximative: 1 = Hiver, 2 = Printemps, 3 = Ete, 4 = Automne
                changement_prix = prix_mois - produit.prix_initial
                facteur_rarete = np.random.uniform(0.9, 1.1)
                facteur_saison = 1 + 0.1 * np.sin(2 * np.pi * (i / 12))
                choc_demande = np.random.uniform(
                    -0.1, 0.1
                )  # Re-simuler la demande aleatoire pour chaque mois
                choc_demande_cumulatif = sum(
                    [np.random.uniform(-0.1, 0.1) for _ in range(i + 1)]
                )

                taux_inflation = taux_inflation_liste[i]

                donnees.append(
                    {
                        "Categorie": nom_categorie,
                        "SKU": produit.sku,
                        "Date": date_vente,
                        "Prix": prix_mois,
                        "PrixInitial": produit.prix_initial,
                        "DateLancement": date_lancement,
                        "TempsDepuisLancement": temps_depuis_lancement,
                        "CategorieAgeProduit": categorie_age_produit,
                        "NombreDeJoursEnRayon": nombre_de_jours_en_rayon,
                        "Saison": saison,
                        "TauxInflation": taux_inflation,
                        "Elasticite": produit.elasticite,
                        "ChocDemande": choc_demande,
                        "ChangementPrix": changement_prix,
                        "FacteurRarete": facteur_rarete,
                        "FacteurSaison": facteur_saison,
                        "MarcheAleatoire": prix_mois,
                        "ChocDemandeCumulatif": choc_demande_cumulatif,
                        "AgeProduitEnJours": age_produit_en_jours,
                    }
                )

    return pd.DataFrame(donnees)


# Fonction pour tracer les SKUs pour une catégorie donnée en utilisant Seaborn
def tracer_skus_pour_categorie(df, categorie, nombre_skus=3, skus_selectionnes=None):
    """
    Trace les fluctuations de prix pour les SKUs dans une catégorie donnée.

    Paramètres:
        df (pd.DataFrame): Le DataFrame contenant les données des SKUs.
        categorie (str): La catégorie de produit.
        nombre_skus (int): Nombre de SKUs à tracer (utilisé si skus_selectionnes est None).
        skus_selectionnes (list): Liste des SKUs sélectionnés manuellement à tracer.
    """
    if skus_selectionnes is None:
        # Sélectionner les SKUs aléatoirement si non sélectionnés manuellement
        skus = (
            df[df["Categorie"] == categorie]["SKU"]
            .drop_duplicates()
            .sample(nombre_skus)
        )
    else:
        # Utiliser les SKUs sélectionnés manuellement
        skus = skus_selectionnes

    plt.figure(figsize=(14, 10))

    for sku in skus:
        sku_data = df[df["SKU"] == sku]
        sns.lineplot(data=sku_data, x="Date", y="Prix", label=sku, marker="o")

    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.title(f"Évolution du prix pour les SKUs dans la catégorie {categorie}")
    plt.legend()
    plt.show()
