import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib

# =========================================================
# CHARGEMENT DU MODÈLE ET DU PIPELINE
# =========================================================
model = joblib.load('model_absenteisme_rf.pkl')
encoding_map = joblib.load('reason_encoding_map.pkl')
moyenne_globale = joblib.load('moyenne_globale_reason.pkl')
variables_selectionnees = joblib.load('variables_selectionnees.pkl')

# Libellés des motifs d'absence (codes CID simplifiés — les plus fréquents dans le dataset)
MOTIFS_ABSENCE = {
    0: "Aucun motif spécifique / Justification administrative",
    1: "Maladies infectieuses et parasitaires",
    6: "Maladies du système nerveux",
    10: "Maladies de l'appareil respiratoire",
    13: "Maladies de la peau",
    19: "Blessures, empoisonnements",
    22: "Consultation de suivi",
    23: "Consultation médicale (motif général)",
    25: "Certificat médical (fièvre non spécifiée)",
    26: "Absence non justifiée",
    27: "Motif physiothérapeutique",
    28: "Examen dentaire",
}


class AppAbsenteisme:
    def __init__(self, root):
        self.root = root
        self.root.title("Aide à la décision RH — Prédiction d'absentéisme")
        self.root.geometry("650x750")

        canvas_scroll = tk.Canvas(root)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas_scroll.yview)
        self.frame = ttk.Frame(canvas_scroll, padding=15)

        self.frame.bind("<Configure>", lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.create_window((0, 0), window=self.frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        canvas_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(self.frame, text="Profil de l'employé", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 15), sticky="w")

        self.champs = {}
        self._ajouter_champ_liste("Motif d'absence prévu", "reason",
                                   list(MOTIFS_ABSENCE.values()), row=1)
        self._ajouter_champ_numerique("Mois (1-12)", "month", "7", row=2)
        self._ajouter_champ_numerique("Jour de la semaine (2=lundi ... 6=vendredi)", "day", "3", row=3)
        self._ajouter_champ_numerique("Frais de transport", "transport", "200", row=4)
        self._ajouter_champ_numerique("Distance domicile-travail (km)", "distance", "20", row=5)
        self._ajouter_champ_numerique("Ancienneté (années de service)", "service_time", "10", row=6)
        self._ajouter_champ_numerique("Âge", "age", "35", row=7)
        self._ajouter_champ_numerique("Charge de travail moyenne/jour", "workload", "270", row=8)
        self._ajouter_champ_numerique("Objectif atteint (%)", "hit_target", "95", row=9)
        self._ajouter_champ_numerique("Nombre d'enfants", "son", "1", row=10)
        self._ajouter_champ_numerique("Taille (cm)", "height", "170", row=11)
        self._ajouter_champ_numerique("Poids (kg)", "weight", "70", row=12)

        ttk.Button(self.frame, text="Prédire le risque d'absentéisme", command=self.predire).grid(
            row=13, column=0, columnspan=2, pady=20, sticky="ew")

        self.resultat_label = ttk.Label(self.frame, text="", font=("Arial", 12, "bold"), wraplength=550)
        self.resultat_label.grid(row=14, column=0, columnspan=2, pady=10)

        self.detail_label = ttk.Label(self.frame, text="", font=("Arial", 10), wraplength=550, justify="left")
        self.detail_label.grid(row=15, column=0, columnspan=2, pady=5)

    def _ajouter_champ_numerique(self, label, cle, valeur_defaut, row):
        ttk.Label(self.frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entree = ttk.Entry(self.frame, width=15)
        entree.insert(0, valeur_defaut)
        entree.grid(row=row, column=1, sticky="w", pady=4)
        self.champs[cle] = entree

    def _ajouter_champ_liste(self, label, cle, options, row):
        ttk.Label(self.frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
        combo = ttk.Combobox(self.frame, values=options, width=35, state="readonly")
        combo.current(0)
        combo.grid(row=row, column=1, sticky="w", pady=4)
        self.champs[cle] = combo

    def predire(self):
        try:
            month = int(self.champs["month"].get())
            day = int(self.champs["day"].get())
            transport = float(self.champs["transport"].get())
            distance = float(self.champs["distance"].get())
            service_time = float(self.champs["service_time"].get())
            age = float(self.champs["age"].get())
            workload = float(self.champs["workload"].get())
            hit_target = float(self.champs["hit_target"].get())
            son = float(self.champs["son"].get())
            height = float(self.champs["height"].get())
            weight = float(self.champs["weight"].get())

            # Retrouve le code du motif choisi dans le menu déroulant
            motif_choisi_texte = self.champs["reason"].get()
            code_motif = next(k for k, v in MOTIFS_ABSENCE.items() if v == motif_choisi_texte)

            # Ingénierie temporelle (identique au notebook)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            day_sin = np.sin(2 * np.pi * day / 7)
            day_cos = np.cos(2 * np.pi * day / 7)

            # Target encoding du motif (avec repli sur la moyenne globale si motif inconnu)
            reason_encoded = encoding_map.get(code_motif, moyenne_globale)

            # BMI calculé
            bmi = weight / ((height / 100) ** 2)

            # Construction du vecteur dans l'ordre EXACT attendu par le modèle
            valeurs = {
                'Reason_encoded': reason_encoded,
                'Work load Average/day ': workload,
                'Hit target': hit_target,
                'Month_cos': month_cos,
                'Month_sin': month_sin,
                'Transportation expense': transport,
                'Day_sin': day_sin,
                'Service time': service_time,
                'Age': age,
                'Son': son,
                'Distance from Residence to Work': distance,
                'Body mass index': bmi,
                'Day_cos': day_cos,
                'Height': height,
                'Weight': weight,
            }

            X_input = np.array([[valeurs[col] for col in variables_selectionnees]])

            proba = model.predict_proba(X_input)[0]
            prediction = model.predict(X_input)[0]

            if prediction == 1:
                self.resultat_label.config(
                    text=f"⚠️ Risque d'absence LONGUE (≥4h) — probabilité : {proba[1]:.1%}",
                    foreground="#c0392b")
            else:
                self.resultat_label.config(
                    text=f"✅ Risque d'absence COURTE (<4h) — probabilité : {proba[0]:.1%}",
                    foreground="#27ae60")

            self.detail_label.config(
                text=f"Détail : probabilité absence courte = {proba[0]:.1%} | "
                     f"probabilité absence longue = {proba[1]:.1%}\n"
                     f"IMC calculé : {bmi:.1f}\n\n"
                     f"Ceci est un outil d'aide à la décision, pas un diagnostic définitif. "
                     f"Il doit être utilisé en complément du jugement RH.")

        except (ValueError, StopIteration) as e:
            self.resultat_label.config(text="Erreur : vérifie que tous les champs sont remplis correctement.",
                                        foreground="#c0392b")
            self.detail_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = AppAbsenteisme(root)
    root.mainloop()