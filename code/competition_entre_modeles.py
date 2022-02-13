from jeu_du_morpion import obtenir_action_agent
from alphazero import AlphaZero
from morpion import Morpion
import torch
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Processeur : {device}")


def partie_entre_modeles(nom_fichier_modele1, nom_fichier_modele2, utiliser_agent_heuristique=False):
    jeu = Morpion()  # On instancie un jeu
    taille_entree = jeu.taille_grille_et_tour  # L'entrée du réseau est de taille 9 + 1 (taille de la grille + tour)
    taille_sortie = jeu.taille_grille  # Il y a 9 actions possibles
    taille_couches_cachees = 128  # Taille des couches cachées
    agent1 = None
    agent2 = None

    # L'agent est du type du modèle choisit
    agent1 = AlphaZero(taille_entree, taille_couches_cachees, taille_sortie)
    # On charge le modèle préenregistré
    agent1.load_state_dict(torch.load(nom_fichier_modele1, map_location=device))

    agent2 = AlphaZero(taille_entree, taille_couches_cachees, taille_sortie)
    # On charge le modèle préenregistré
    agent2.load_state_dict(torch.load(nom_fichier_modele2, map_location=device))
    # On passe en mode évaluation
    agent1.eval()
    agent2.eval()
    nbre_coup = 0
    # On fait la partie entre les deux agents
    etat, recompense, fini = jeu.reinitialiser()  # On initialise la partie
    while not fini:  # Tant que la partie n'est pas finie
        # On récupère l'action du premier agent
        action_agent1 = obtenir_action_agent(agent1, jeu, etat, verbose=False)

        # On effectue l'action puis on passe au tour suivant
        etat, recompense, fini = jeu.effectuer_pas(action_agent1)
        nbre_coup += 1

        jeu.afficher_grille()  # On affiche de nouveau la grille
        print(f"recompense : {recompense}")
        print("\n")

        if not fini:
            # On récupère l'action du deuxième agent
            action_agent2 = obtenir_action_agent(agent2, jeu, etat, verbose=False)
            # On effectue l'action puis on passe au tour suivant
            etat, recompense, fini = jeu.effectuer_pas(action_agent2)
            nbre_coup += 1

            jeu.afficher_grille()  # On affiche de nouveau la grille
            print(f"recompense : {recompense}")
            print("\n")

    if recompense == 0:
        print("Egalité")
    elif recompense == 1:
        print(f"L'agent 1 a gagné")
    else:
        print(f"L'agent 2 a gagné")
    return recompense, nbre_coup


if __name__ == "__main__":
    # On charge un jeu avec un modèle d'agent enregistré et où le vrai joueur joue en premier
    duels = np.array([[0, 1], [0, 2], [1, 0], [2, 0], [1, 2], [2, 1]])
    noms_agents = ["faible", "moyen", "fort"]
    tableau_resultats = pd.DataFrame({"duel": [f"{noms_agents[0]} vs {noms_agents[1]}",
                                               f"{noms_agents[0]} vs {noms_agents[2]}",
                                               f"{noms_agents[1]} vs {noms_agents[0]}",
                                               f"{noms_agents[2]} vs {noms_agents[0]}",
                                               f"{noms_agents[1]} vs {noms_agents[2]}",
                                               f"{noms_agents[2]} vs {noms_agents[1]}"]})
    gagnants = [0, 0, 0, 0, 0, 0]
    nbrecoups = np.zeros(6)
    for i in range(6):
        duel = duels[i]
        premier_joueur, deuxieme_joueur = noms_agents[duel[0]], noms_agents[duel[1]]
        issue, nbrecoups[i] = partie_entre_modeles(f"../models/tictactoe_agent{premier_joueur}.pt",
                                                   f"../models/tictactoe_agent{deuxieme_joueur}.pt")
        if issue == 1:
            gagnants[i] = premier_joueur
        elif issue == -1:
            gagnants[i] = deuxieme_joueur
        else:
            gagnants[i] = "égalité"
    tableau_resultats["resultat"] = pd.Series(gagnants)
    tableau_resultats["nombre de coups"] = pd.Series(nbrecoups)
    print(tableau_resultats)
