from alphazero import AlphaZero
from morpion import Morpion
import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Processeur : {device}")


def charger_et_jouer(nom_fichier_modele, tour_agent=1, utiliser_agent_heuristique=False):
    jeu = Morpion()  # On instancie un jeu
    taille_entree = jeu.taille_grille_et_tour  # L'entrée du réseau est de taille 9 + 1 (taille de la grille + tour)
    taille_sortie = jeu.taille_grille  # Il y a 9 actions possibles
    taille_couches_cachees = 128  # Taille des couches cachées

    if utiliser_agent_heuristique:
        agent = None
    else:  # Si on choisit un agent entraîné
        # L'agent est du type du modèle choisit
        agent = AlphaZero(taille_entree, taille_couches_cachees, taille_sortie)
        agent.load_state_dict(torch.load(nom_fichier_modele, map_location=device))  # On charge le modèle préenregistré
    # On passe en mode évaluation
    agent.eval()
    # On joue avec l'agent
    jouer_avec_agent(agent, tour_agent=tour_agent, utiliser_agent_heuristique=utiliser_agent_heuristique, verbose=True)


def obtenir_action_agent(agent, jeu, etat, verbose=False):
    masque = jeu.obtenir_actions_autorisees()  # On récupère toutes les actions autorisées
    # On crée un tenseur contenant l'état de la grille
    tenseur_etat = torch.tensor(etat, device=device, dtype=torch.float)
    prediction = agent(tenseur_etat)  # On passe l'état à travers le modèle
    probas = prediction[:-1]  # On récupère un vecteur de probabilité pour chaque action possible
    # On récupère la valeur associée à l'état actuel (récompense espérée en continuant dans cette voie)
    value = prediction[-1].detach().numpy()
    prob = probas.detach().numpy()
    prob = prob * masque  # On ne garde que les actions légales
    prob = prob / np.sum(prob)  # On s'assure de sommer à 1
    action = np.argmax(prob)  # On choisit l'action de plus grande proba

    if verbose:
        print(f"Valeur : {value}, Probabilités : {prob}, Action : {action}")

    return action


def jouer_avec_agent(agent, tour_agent=1, utiliser_agent_heuristique=False, verbose=False):
    jeu = Morpion()
    etat, recompense, fini = jeu.reinitialiser()  # On initialise la partie
    jeu.afficher_grille()  # On affiche la grille
    while not fini:  # Tant que la partie n'est pas finie
        if tour_agent == 1:  # Si c'est l'agent qui joue en premier

            if utiliser_agent_heuristique:  # Si on a choisi l'agent heuristique
                action_agent = jeu.obtenir_action_heuristique()
            else:  # Si on a choisi l'agent entraîné avec le modèle
                # On récupère l'action de l'agent
                action_agent = obtenir_action_agent(agent, jeu, etat, verbose=verbose)

            # On effectue l'action puis on passe au tour suivant
            etat, recompense, fini = jeu.effectuer_pas(action_agent)
            jeu.afficher_grille()  # On affiche de nouveau la grille

            if not fini:  # Si la partie n'est pas finie après l'action de l'agent
                action = int(input("Saisir une case où jouer (0-8) : "))  # On récupère l'action du vrai joueur

                # On continue de demander une action jusqu'à ce qu'elle soit légale
                while not jeu.verifier_action_autorisee(action):
                    print("Action interdite")
                    action = int(input("Saisir une case où jouer (0-8): "))

                etat, recompense, fini = jeu.effectuer_pas(action)  # On effectue l'action et on met à jour le jeu
                jeu.afficher_grille()  # On affiche la grille

        elif tour_agent == 2:  # Si c'est au vrai joueur de commencer
            action = int(input("Saisir une case où jouer (0-8): "))

            while not jeu.verifier_action_autorisee(action):
                print("Action interdite")
                action = int(input("Saisir une case où jouer (0-8): "))

            etat, recompense, fini = jeu.effectuer_pas(action)
            jeu.afficher_grille()
            if not fini:
                if utiliser_agent_heuristique:
                    action_agent = jeu.obtenir_action_heuristique()
                else:
                    action_agent = obtenir_action_agent(agent, jeu, etat, verbose=verbose)
                etat, recompense, fini = jeu.effectuer_pas(action_agent)
                jeu.afficher_grille()

        else:
            raise Exception("Valeur invalide pour le tour de l'agent. "
                            "Choisir 1 si l'agent commence ou 2 pour commencer.")

    if recompense == 0:  # Si on finit sur une récompense de 0, c'est qu'il y a match nul
        print("Égalité")
    # Si c'est au tour de l'agent et qu'il gagne ou que c'est au tour du vrai joueur et qu'il perd, l'agent gagne
    elif (recompense == 1 and tour_agent == 1) or (recompense == -1 and tour_agent == 2):
        print("L'agent a gagné")
    else:  # Sinon, c'est que c'est le vrai joueur qui a gagné
        print("Vous avez gagné !")


if __name__ == "__main__":
    # On charge un jeu avec un modèle d'agent enregistré et où le vrai joueur joue en premier
    charger_et_jouer("../models/tictactoe_agentfort60.pt", tour_agent=2)
