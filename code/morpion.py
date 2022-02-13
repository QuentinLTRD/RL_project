import numpy as np

class Morpion:
    def __init__(self):
        self.etat_jeu = np.zeros((3, 3))
        self.tour_joueur = 1
        self.taille_grille = 9  # 9 positions où jouer
        self.taille_grille_et_tour = 9 + 1  # 9 positions + le joueur dont c'est le tour

    def reinitialiser(self):
        self.etat_jeu = np.zeros((3, 3))
        self.tour_joueur = 1
        etat = self.etat_jeu.flatten()
        etat = np.append(etat, [self.tour_joueur])
        etat = etat.astype(np.float32)
        return etat, 0, False

    def effectuer_pas(self, action):
        # L'action doit être un nombre entre 0 et 8
        if not self.verifier_action_autorisee(action):
            print("Action interdite jouée")
            print(f"Action : {action}")
            self.afficher_grille()
            print("")
            raise Exception("Action interdite")

        i = action // 3
        j = action % 3

        self.etat_jeu[i, j] = self.tour_joueur  # On place un 1 ou un -1 à l'emplacement choisi en fonction du joueur
        self.tour_joueur = -self.tour_joueur  # On passe au tour de l'autre joueur
        fini, issue = self.verifier_partie_finie()

        etat = self.etat_jeu.flatten()
        etat = np.append(etat, [self.tour_joueur])

        return etat, issue, fini

    def afficher_grille(self):
        print(self.etat_jeu)

    def verifier_partie_finie(self):
        # On vérifie les lignes
        for i in range(3):
            egal = True
            for j in range(1, 3):
                egal = egal and self.etat_jeu[i, 0] == self.etat_jeu[i, j]
            if egal and self.etat_jeu[i, 0] != 0:
                fini = True
                resultat = self.etat_jeu[i, 0]
                return fini, resultat

        # On vérifie les colonnes
        for j in range(3):
            egal = True
            for i in range(1, 3):
                egal = egal and self.etat_jeu[0, j] == self.etat_jeu[i, j]
            if egal and self.etat_jeu[0, j] != 0:
                fini = True
                resultat = self.etat_jeu[0, j]
                return fini, resultat

        # On vérifie la diagonale qui part d'en haut à gauche
        egal = True
        for i in range(1, 3):
            j = i
            egal = egal and self.etat_jeu[0, 0] == self.etat_jeu[i, j]

        if egal and self.etat_jeu[0, 0] != 0:
            fini = True
            resultat = self.etat_jeu[0, 0]
            return fini, resultat

        # On vérifie la diagonale qui part d'en haut à droite
        egal = True
        for i in range(1, 3):
            j = 2 - i
            egal = egal and self.etat_jeu[0, 2] == self.etat_jeu[i, j]

        if egal and self.etat_jeu[2, 0] != 0:
            fini = True
            resultat = self.etat_jeu[2, 0]
            return fini, resultat

        # Si tout le plateau est rempli, la partie est finie
        # On vérifie tout de même qu'il ne reste pas un emplacement où jouer
        for i in range(3):
            for j in range(3):
                if self.etat_jeu[i, j] == 0:
                    return False, 0
        # Tout le plateau est rempli, mais personne n'a gagné
        return True, 0

    def verifier_action_autorisee(self, action):
        if action < 0:
            return False
        if action > 8:
            return False
        i = action // 3
        j = action % 3
        return self.etat_jeu[i, j] == 0

    def obtenir_actions_autorisees(self):
        masque = np.zeros(9)
        for i in range(9):
            masque[i] = 1 if self.verifier_action_autorisee(i) else 0
        return masque

    def copier_jeu(self):
        copie_jeu = Morpion()
        copie_jeu.etat_jeu = np.copy(self.etat_jeu)
        copie_jeu.tour_joueur = self.tour_joueur
        return copie_jeu

    # Fonction permettant de trouver la meilleure action pour le joueur actuel, basée sur des heuristiques simples
    def obtenir_action_heuristique(self):
        # On vérifie si l'on peut gagner directement
        for i in range(9):
            jeu_test = self.copier_jeu()
            if jeu_test.verifier_action_autorisee(i):
                etat, recompense, fini = jeu_test.effectuer_pas(i)
                if fini and recompense == self.tour_joueur:
                    return i

        # On vérifie si l'autre joueur peut trouver un carré dans lequel il gagne et le bloquer si c'est le cas
        for i in range(9):
            jeu_test = self.copier_jeu()
            if jeu_test.verifier_action_autorisee(i):
                jeu_test.etat_jeu[i // 3, i % 3] = -self.tour_joueur
                fini, issue = jeu_test.verifier_partie_finie()
                if fini:
                    return i

        # Si possible, on se place au centre
        if self.verifier_action_autorisee(4):
            return 4

        # Prendre un coin si l'un est libre
        coins = [0, 2, 6, 8]
        for coin in coins:
            if self.verifier_action_autorisee(coin):
                return coin

        # Prendre un côté si l'un est libre
        cotes = [1, 3, 5, 7]
        for cote in cotes:
            if self.verifier_action_autorisee(cote):
                return cote
