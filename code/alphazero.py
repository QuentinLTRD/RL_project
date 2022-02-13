import torch
import numpy as np
import time
from torch.nn import Linear, BatchNorm1d
from torch.nn.functional import relu, softmax
from torch import tanh
from morpion import Morpion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Processeur : {device}")


# Le réseau de neurones entraîné et utilisé
class AlphaZero(torch.nn.Module):
    def __init__(self, dimension_entree, dimension_couche_cachee, dimension_sortie):
        super().__init__()
        self.dimension_entree = dimension_entree
        self.dimension_sortie = dimension_sortie
        self.couche_entree = Linear(dimension_entree, dimension_couche_cachee)
        self.couches_residuelles = torch.nn.ModuleList(
            [Linear(dimension_couche_cachee, dimension_couche_cachee),
             Linear(dimension_couche_cachee, dimension_couche_cachee),
             Linear(dimension_couche_cachee, dimension_couche_cachee)])

        self.batch_norms = torch.nn.ModuleList([BatchNorm1d(dimension_couche_cachee) for _ in self.couches_residuelles])
        self.policy = Linear(dimension_couche_cachee, dimension_sortie)
        self.value = Linear(dimension_couche_cachee, 1)

    def forward(self, x):
        if len(x.shape) == 1:  # Si le tenseur en entrée est de longueur 1
            x = x.unsqueeze(0)  # On ajoute une dimension en position 0 (on met le tableau dans un tableau)

        hidden = relu(self.couche_entree(x))
        for i, l in enumerate(self.couches_residuelles):  # On passe à travers les couches cachées
            hidden = l(hidden) + hidden  # Bloc résiduel
            hidden = self.batch_norms[i](hidden)
            hidden = relu(hidden)
        p = softmax(self.policy(hidden), dim=-1)  # On récupère les probabilités pour chaque action
        _v = self.value(hidden)
        v = tanh(_v)  # On prédit une valeur pour cet état
        return torch.cat((p, v), dim=1).squeeze()


# Nœud du MCTS
class Noeud:
    def __init__(self, etat, modele, noeud_parent):
        self.c = 1  # Hyper-paramètre
        self.etat = np.copy(etat)  # État de la grille
        self.z = None  # Paramètre non utilisé
        self.parent = noeud_parent  # Noeud parent
        self.joueur = 1 if noeud_parent is None else -noeud_parent.joueur  # +1 si c'est le premier joueur, -1 si second

        x = torch.tensor(self.etat, device=device, dtype=torch.float32)  # Tenseur contenant l'état

        prediction = modele(x)  # On passe l'état à travers le modèle pour avoir une prédiction
        # On crée les données pour l'entraînement
        self.proba_init = prediction[:-1].detach().cpu().numpy()  # On récupère les probas des actions
        self.value = prediction[-1].detach().cpu().numpy()  # et la valeur associée à l'état courant

        nb_actions = prediction.size()[0] - 1
        self.Q = np.zeros(nb_actions)  # .5 * np.ones(nb_actions)
        self.actions_prises = np.zeros(nb_actions)
        self.derniere_action = -1  # Utilisé quand on remonte dans l'arbre pour mettre à jour les valeurs de Q


def chercher_action(noeud_courant, dict_noeuds, agent, jeu):  # On récupère la prochaine action
    # Une borne supérieure de confiance pour chaque action
    ucb = noeud_courant.Q + noeud_courant.c * noeud_courant.proba_init * np.sqrt(np.sum(noeud_courant.actions_prises)) \
          / (1 + noeud_courant.actions_prises) + 1e-6  # On ajoute une petite valeur pour la stabilité numérique

    masque = jeu.obtenir_actions_autorisees()
    ucb_masque = masque * ucb

    ucb_masque[ucb_masque == 0] = np.nan  # On transforme les 0 en NaN pour qu'ils soient ignorés par argmax
    action = np.nanargmax(ucb_masque)  # On choisit l'action de borne maximale

    if np.random.random() < .2:  # On choisit une action autorisée de manière aléatoire
        probas = np.ones_like(ucb_masque)
        probas = probas * masque
        probas = probas / np.sum(probas)
        action = np.random.choice(list(range(jeu.taille_grille)), p=probas)

    noeud_courant.derniere_action = action
    noeud_courant.actions_prises[action] += 1

    prochain_etat, recompense, fini = jeu.effectuer_pas(action)  # On effectue l'action trouvée

    # Si le jeu finit
    if fini:
        noeud = noeud_courant  # On peut mettre à jour le nœud
        propager_valeurs_dans_arbre(noeud, recompense)  # Et propager les valeurs dans l'arbre
        return

    # Si le jeu n'est pas fini
    else:
        clef = prochain_etat.tobytes()
        if clef in dict_noeuds:  # Si le prochain état est dans l'arbre
            noeud_suivant = dict_noeuds[clef]
            # On appelle récursivement la fonction afin d'atteindre une fin de jeu
            chercher_action(noeud_suivant, dict_noeuds, agent, jeu)
            return
        else:
            # On crée un nouveau nœud et on remonte les valeurs dans l'arbre
            creer_noeud_dans_arbre(prochain_etat, dict_noeuds, agent, noeud_courant)
            return


def propager_valeurs_dans_arbre(noeud_feuille, value):
    noeud = noeud_feuille
    # On met à jour Q
    q_new = noeud.Q[noeud.derniere_action] * (noeud.actions_prises[noeud.derniere_action] - 1) + value * noeud.joueur
    q_new /= noeud.actions_prises[noeud.derniere_action]
    noeud.Q[noeud.derniere_action] = q_new

    if noeud_feuille.parent is None:  # Si on atteint la racine, on peut arrêter
        return
    else:
        noeud = noeud_feuille.parent  # Sinon, on récupère le nœud père
        propager_valeurs_dans_arbre(noeud, value)  # Et on propage la valeur au père


def creer_noeud_dans_arbre(etat, dict_noeuds, agent, dernier_noeud):
    nouveau_noeud = Noeud(etat, agent, dernier_noeud)
    valeur = nouveau_noeud.value

    # On met à jour les valeurs dans l'arbre
    propager_valeurs_dans_arbre(dernier_noeud, valeur)

    # On enregistre le graphe dans un dictionnaire
    clef = etat.tobytes()
    dict_noeuds[clef] = nouveau_noeud
    return nouveau_noeud


def simuler_jeu(jeu, agent, nb_etapes_recherche):
    liste_etats = []  # Liste destinée à contenir les états
    liste_pi = []  # Liste destinée à contenir les probabilités de chaque action
    liste_issue = []  # Liste destinée à contenir l'issue finale d'un jeu

    etat, recompense, fini = jeu.reinitialiser()

    racine = Noeud(etat, agent, None)
    dict_noeuds = {racine.etat.tobytes(): racine}

    noeud_courant = racine

    while not fini:
        etat_courant = etat
        clef = etat.tobytes()
        if clef in dict_noeuds:
            noeud_courant = dict_noeuds[clef]
        else:
            noeud_courant = creer_noeud_dans_arbre(etat_courant, dict_noeuds, agent, noeud_courant)

        noeud_courant.parent = None
        for n in range(nb_etapes_recherche):
            copie_jeu = jeu.copier_jeu()  # On copie le jeu, car il va être modifié
            chercher_action(noeud_courant, dict_noeuds, agent, copie_jeu)

        # On normalise les actions pour avoir des probabilités apprises durant le MCTS
        masque_actions_autorisees = jeu.obtenir_actions_autorisees()
        pi = noeud_courant.actions_prises * masque_actions_autorisees
        pi = pi / np.sum(pi)
        copie_pi = pi

        liste_etats.append(etat_courant)
        liste_pi.append(copie_pi)
        liste_issue.append(0)  # Contiendra l'issue du jeu

        if np.random.random() < .2:  # On sélectionne une action de manière aléatoire
            probas = np.ones_like(pi)
            probas = probas * masque_actions_autorisees
            probas = probas / np.sum(probas)
            action = np.random.choice(list(range(jeu.taille_grille)), p=probas)
        else:  # Ou on sélectionne une action proportionnelle aux probabilités calculées avec le MCTS
            action = np.random.choice(list(range(jeu.taille_grille)), p=pi)

        # recompense vaut +1 si le premier joueur gagner, -1 si c'est le second, 0 s'il y a égalité
        etat, recompense, fini = jeu.effectuer_pas(action)

        if fini:
            liste_etats.append(etat)
            liste_pi.append(np.zeros_like(copie_pi))
            liste_issue.append(recompense)

    for i in range(len(liste_issue)):
        liste_issue[i] = recompense  # On peut mettre à jour l'issue du jeu

    return liste_etats, liste_pi, liste_issue


# Fonction permettant de générer les données d'entraînement
def generer_donnees_entrainement(jeu, nb_jeux, nb_etapes_recherche, agent):
    liste_etats, liste_pi, liste_issue = simuler_jeu(jeu, agent, nb_etapes_recherche)

    # On ne tient pas compte du UserWarning qui n'apporte pas de réelle différence sur le temps de calcul
    tenseur_etats = torch.tensor(liste_etats, device=device).to(dtype=torch.float)
    tenseur_probas = torch.tensor(liste_pi, device=device).to(dtype=torch.float)
    tenseur_issues = torch.tensor(liste_issue, device=device).to(dtype=torch.float)

    for _ in range(nb_jeux - 1):
        liste_etats, liste_pi, liste_issue = simuler_jeu(jeu, agent, nb_etapes_recherche)
        tenseur_etats_i = torch.tensor(liste_etats, device=device).to(dtype=torch.float)
        tenseur_probas_i = torch.tensor(liste_pi, device=device).to(dtype=torch.float)
        tenseur_issue_i = torch.tensor(liste_issue, device=device).to(dtype=torch.float)

        tenseur_etats = torch.cat((tenseur_etats, tenseur_etats_i))
        tenseur_probas = torch.cat((tenseur_probas, tenseur_probas_i))
        tenseur_issues = torch.cat((tenseur_issues, tenseur_issue_i))

    print(f"Nouveaux états générés : {tenseur_etats.size(0)}")
    return tenseur_etats, tenseur_probas, tenseur_issues


# Fonction permettant d'améliorer le modèle avec les jeux simulés
def ameliorer_modele(modele, donnees_entrainement, nb_epoques, optimiseur, taille_batch=64):
    loss_moyenne = None
    etats = donnees_entrainement[0]
    probas = donnees_entrainement[1]
    recompense = donnees_entrainement[2]
    value_loss = 0
    policy_loss = 0

    for i in range(nb_epoques):
        shuffle = torch.randperm(etats.size(0))
        etats = etats[shuffle]
        probas = probas[shuffle]
        recompense = recompense[shuffle]

        for b in range(0, etats.size(0), taille_batch):
            fin = b + taille_batch if (b + taille_batch < etats.size(0)) else (etats.size(0))
            if fin - b < 2:  # On saute les derniers points s'il y en a
                continue

            pred = modele(etats[b: fin, :])
            prob = pred[:, :-1] + 1e-6  # On rajoute 1e-6 pour ne pas prendre log(0)
            val = pred[:, -1]

            value_loss = torch.mean((val - recompense[b:fin]) * (val - recompense[b:fin]))
            policy_loss = -torch.mean(torch.sum(probas[b:fin] * torch.log(prob), dim=-1))
            loss_moyenne = value_loss + policy_loss

            if torch.isnan(loss_moyenne):
                print("NaN obtenus")

            optimiseur.zero_grad()
            loss_moyenne.backward()
            optimiseur.step()
        print(f"Époque : {i + 1}, Loss : {loss_moyenne}")

    return loss_moyenne.detach().cpu().numpy(), value_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()


def obtenir_action_agent(agent, jeu, etat, greedy=False, verbose=False):
    masque = jeu.obtenir_actions_autorisees()
    tenseur_etats = torch.tensor(etat, device=device, dtype=torch.float)
    prediction = agent(tenseur_etats)
    probas = prediction[:-1]
    value = prediction[-1].detach().cpu().numpy()
    prob = probas.detach().cpu().numpy() + 1e-6
    prob = prob * masque
    prob = prob / np.sum(prob)
    if greedy:
        action = np.argmax(prob)
    else:
        action = np.random.choice(list(range(9)), p=prob)

    if verbose:
        print(f" Valeur : {value}, Action : {action}")
        print(f"{etat[0]} {etat[1]} {etat[2]} || {prob[0]:.3f} {prob[1]:.3f} {prob[2]:.3f}")
        print(f"{etat[3]} {etat[4]} {etat[5]} || {prob[3]:.3f} {prob[4]:.3f} {prob[5]:.3f}")
        print(f"{etat[6]} {etat[7]} {etat[8]} || {prob[6]:.3f} {prob[7]:.3f} {prob[8]:.3f}")

    return action


def jouer_contre_heuristique(jeu, agent, tour_agent=1, afficher=False, verbose=False):
    etat, r, fini = jeu.reinitialiser()
    recompense = 0
    while not fini:
        if tour_agent == 1:  # Si c'est l'agent qui commence
            agent_action = obtenir_action_agent(agent, jeu, etat, greedy=True, verbose=False)
            etat, recompense, fini = jeu.effectuer_pas(agent_action)
            if afficher:
                jeu.afficher_grille()
            if not fini:
                action = jeu.obtenir_action_heuristique()
                etat, recompense, fini = jeu.effectuer_pas(action)
                if afficher:
                    jeu.afficher_grille()
        elif tour_agent == 2:  # Si c'est l'autre joueur qui commence
            action = jeu.obtenir_action_heuristique()
            etat, recompense, fini = jeu.effectuer_pas(action)
            if afficher:
                jeu.afficher_grille()
            if not fini:
                agent_action = obtenir_action_agent(agent, jeu, etat, greedy=True, verbose=False)
                etat, recompense, fini = jeu.effectuer_pas(agent_action)
                if afficher:
                    jeu.afficher_grille()
        else:
            raise Exception("Valeur invalide pour le tour de l'agent. "
                            "Choisir 1 si l'agent commence ou 2 pour commencer.")

    if verbose:
        if recompense == 0:
            print("Égalité")
        elif recompense == 1 and tour_agent == 1:
            print("L'agent a gagné")
        else:
            print("L'heuristique a gagné")

    return recompense


# Fonction permettant l'entraînement
def main():
    jeu = Morpion()
    taille_entree = jeu.taille_grille_et_tour
    taille_sortie = jeu.taille_grille
    taille_couches_cachees = 128
    agent = AlphaZero(taille_entree, taille_couches_cachees, taille_sortie)

    agent.to(device)
    print(agent)
    agent.eval()
    print(f"Paramètres : {sum([p.nelement() for p in agent.parameters()])}")
    liste_iterations = [0, 15, 60]  # Le nombre de fois où l'on crée des données et entraîne le modèle
    liste_nb_jeux = [0, 20,
                     40]  # Jouer un certain nombre de parties pour générer des exemples à chaque itération (batch)
    liste_nb_etapes_recherche = [0, 20, 40]  # Pour chaque étape de chaque partie, considérer autant d'issues
    lsite_nb_pas_optimisation = [0, 10, 15]  # Le nombre d'époques pour entraîner le modèle sur des données générées
    noms_agents = ["faible", "moyen", "fort"]
    for k in range(0, 3):
        debut_entrainement = time.time()
        iterations = liste_iterations[k]
        nb_jeux = liste_nb_jeux[k]
        nb_etapes_recherche = liste_nb_etapes_recherche[k]
        nb_pas_optimisation = lsite_nb_pas_optimisation[k]

        nb_duels = 2  # Le nombre de parties entre l'ancien et le nouveau modèle pour déterminer le meilleur

        learning_rate = 0.01
        optimiseur = torch.optim.Adam(agent.parameters(), lr=learning_rate, weight_decay=0.01)

        taille_replay_buffer = 800
        meilleure_loss = 0.1
        donnees_entrainement = None
        jeu_contre_heuristique = True

        # On ouvre en lecture/écriture le fichier dans lequel on va enregistrer les données d'entraînement
        with open(f"../models/saved_data{k}.csv", "w+") as f:
            f.write(f"loss totale, value loss, policy loss, victoires agent, victoires heuristique, egalites\n")
            for i in range(iterations):
                print(f"Début de l'itération {i + 1} / {iterations}")

                print(f"Génération des données d'entraînement ...")
                if donnees_entrainement is None:
                    donnees_entrainement = generer_donnees_entrainement(jeu, nb_jeux, nb_etapes_recherche, agent)
                else:  # On génère plus de données si l'agent amélioré ne bat pas l'ancien
                    print("Génération de données additionnelles ...")
                    donnees_supp = generer_donnees_entrainement(jeu, nb_jeux, nb_etapes_recherche, agent)
                    etats = torch.cat((donnees_entrainement[0], donnees_supp[0]))
                    probas = torch.cat((donnees_entrainement[1], donnees_supp[1]))
                    issue = torch.cat((donnees_entrainement[2], donnees_supp[2]))
                    donnees_entrainement = [etats, probas, issue]

                # On tronque les données pour ne garder que les derniers états
                _etats = donnees_entrainement[0][-taille_replay_buffer:, :]
                _probas = donnees_entrainement[1][-taille_replay_buffer:, :]
                _issues = donnees_entrainement[2][-taille_replay_buffer:]

                donnees_entrainement = [_etats, _probas, _issues]

                print(f"Données d'entraînement générées : {donnees_entrainement[0].size(0)} états")

                print(f"Amélioration du modèle ...")
                agent.train()  # On entraîne l'agent
                loss, value_loss, policy_loss = ameliorer_modele(agent, donnees_entrainement,
                                                                 nb_pas_optimisation, optimiseur)
                agent.eval()  # On évalue l'agent

                print(f"Amélioration du modèle terminée, loss totale : {loss},"
                      f" value loss : {value_loss}, policy loss : {policy_loss}")

                if loss < meilleure_loss:
                    torch.save(agent.state_dict(), f"../models/tictactoe_agent_{loss}.pt")  # On enregistre le modèle
                    meilleure_loss = loss

                if jeu_contre_heuristique:
                    agent_gagne_premier = 0
                    agent_gagne_second = 0
                    heuristique_gagne_premier = 0
                    heuristique_gagne_second = 0
                    egalites = 0
                    for j in range(nb_duels // 2):
                        jeu.reinitialiser()
                        r = jouer_contre_heuristique(jeu, agent, tour_agent=1)
                        if r == 0:
                            egalites += 1
                        elif r == 1:
                            agent_gagne_premier += 1
                        elif r == -1:
                            heuristique_gagne_second += 1

                    for j in range(nb_duels // 2):
                        jeu.reinitialiser()
                        r = jouer_contre_heuristique(jeu, agent, tour_agent=2)
                        if r == 0:
                            egalites += 1
                        elif r == 1:
                            heuristique_gagne_premier += 1
                        elif r == -1:
                            agent_gagne_second += 1

                    print(f"Victoires de l'agent : ({agent_gagne_premier},{agent_gagne_second}) ,"
                          f" Victoires de l'heuristique : ({heuristique_gagne_premier},{heuristique_gagne_second}) , "
                          f"Égalités : {egalites}")
                    f.write(f"{loss}, {value_loss}, {policy_loss}, {agent_gagne_premier + agent_gagne_second},"
                            f" {heuristique_gagne_premier + heuristique_gagne_second}, {egalites}\n")

            print("Sauvegarde de l'agent agent")
            torch.save(agent.state_dict(),
                       f"../models/tictactoe_agent{noms_agents[k]}.pt")  # On enregistre le nouveau modèle
            print(f"Meilleure loss : {meilleure_loss}")
        fin_entrainement = time.time()
        temps_entrainement = fin_entrainement - debut_entrainement
        h = temps_entrainement // 3600
        m = (temps_entrainement - h * 3600) // 60
        s = temps_entrainement % 60
        print(f"Temps d'entraînement du modèle {noms_agents[k]} : {h} h {m} m {s} s")

    return agent


if __name__ == "__main__":
    main()

