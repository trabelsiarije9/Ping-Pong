#🏓Ping-Pong

Un remake classique de Pong, développé en Python avec pygame. Jouez à deux sur le même clavier ou affrontez une IA entraînée. Inclut un espace de chat pour discuter en temps réel pendant la partie.


Bienvenue dans ce projet de jeu Ping Pong interactif, développé avec Python, utilisant Pygame pour l’affichage, PyTorch pour l’intelligence artificielle, et un système de chat en réseau local pour la communication entre joueurs.

Ce projet est une démonstration complète mêlant jeu, intelligence artificielle et réseau, dans un esprit à la fois ludique et pédagogique.
Crée par Trabelsi Arije étudiante à l'École Supérieure de l'Économie Numérique.

        ____
🚀 Fonctionnalités Principales :
Ce projet propose une expérience de jeu de ping pong multijoueur avec des options avancées, telles que :
 🎮Mode multijoueur local : Deux personnes peuvent jouer l’une contre l’autre sur le même clavier.
 🤖Mode Joueur contre IA : Une intelligence artificielle entraînée avec l’algorithme DQN (Deep Q-Learning) affronte un joueur humain.
 💬 Système de chat intégré : Les joueurs peuvent échanger des messages via une interface de chat intégrée au jeu.
 🧪 Mode entraînement IA : Possibilité d’entraîner l’IA à jouer seule, en s’améliorant avec le temps.
 📂 Sauvegarde automatique : L’agent IA peut enregistrer et recharger ses progrès depuis un fichier.
🎨 Menu interactif : Un menu d’accueil permet de choisir facilement le mode de jeu ou d’entraînement.

     ____
🧠 Intelligence Artificielle
L’IA repose sur une réseaux de neurones entraîné avec DQN (Deep Q-Learning), un algorithme d’apprentissage par renforcement :
<.> L’agent observe l’environnement (position de la balle, direction, position de la raquette) et choisit une action (monter, descendre, rester).
<.> Elle apprend de ses erreurs et s’améliore au fil des parties, grâce à un système de récompense :
        ✅ +1 si elle renvoie la balle.
        ❌ -1 si elle rate.
<.> L’apprentissage se fait par essais-erreurs : plus l’IA joue, plus elle s’améliore.
<.> Le modèle est stocké dans un fichier .pth et peut être réutilisé sans tout réentraîner.

Ce module est une bonne introduction à l’apprentissage par renforcement pour les passionnés d’IA 🧠 !

             ____
🕹️ Contrôles du Jeu 

🎮 Mode Deux Joueurs
Joueur 1 (gauche) :
W : monter la raquette
S : descendre la raquette

Joueur 2 (droite) :
Flèche Haut (↑) : monter
Flèche Bas (↓) : descendre

🤖 Mode IA vs Joueur
Joueur humain contrôle la raquette de gauche (W / S)
L’IA contrôle automatiquement la raquette de droite

La balle rebondit automatiquement, et le score est comptabilisé à chaque point marqué.

     ___
💬 Système de Chat Intégré
Le jeu dispose d’un système de chat textuel en réseau local :
Active le chat via un bouton dans l’interface (ou une touche dédiée)
Tape ton message, puis appuie sur Entrée pour l’envoyer.
Tous les messages sont affichés dans une boîte de discussion visible pendant le jeu.
Cela permet de communiquer en temps réel sans sortir du jeu, une fonctionnalité idéale pour les parties multijoueurs.

     ____
🧪 Entraîner l’IA
Depuis le menu principal, sélectionne "3. Entraîner l'IA" pour lancer une session d’apprentissage :
L’agent joue contre lui-même, apprend à bloquer la balle et à anticiper.
L’algorithme DQN est utilisé pour améliorer ses décisions à chaque épisode.
Le fichier dqn_pong.pth est mis à jour avec le nouveau modèle.
REMARQUE : L'entraînement peut être long, mais tu verras de vrais progrès ! 

    ___
▶️ Installation & Lancement
1. Installation des dépendances :
bash
pip install pygame torch numpy

2. Lancer le jeu :
bash
python pingpong.py

    ___
🛠️ À faire

 Ajouter un mode en ligne.
 Sauvegarder l’historique des discussions.
 Améliorer l’interface graphique.

  ____
📸 Aperçu 

Début de jeu ! ☝🏼
Cette photo montre le menu du jeu avec trois options principales :
1/Deux joueurs : Jouez avec un ami en utilisant les flèches directionnelles et les touches "W" et "S" pour contrôler vos raquettes.
2/Joueur contre l'IA : Affrontez un agent IA et testez vos compétences contre un adversaire virtuel.
Q/Quitter : Quittez le jeu en fermant l'onglet.
En bas de l'écran, vous trouverez l'indication "Cliquez sur TAB pour ouvrir le chat", vous permettant de discuter avec votre ami pendant la partie
![Capture d'écran 2025-04-14 200427](https://github.com/user-attachments/assets/d8025628-1605-4813-97be-48bd5af9f499)

 En pleine partie !🏓
Cette image montre une session de jeu multijoueur où les joueurs s’affrontent en temps réel. La balle rebondit entre les deux raquettes, le score est comptabilisé automatiquement, et le système de chat permet aux joueurs de communiquer pendant la partie.
![Capture d'écran 2025-04-14 201507](https://github.com/user-attachments/assets/2e363d79-87c2-49fe-968e-e676d50f52ce)

Ça joue et ça jase ! 🤗
Cette photo montre une partie en cours où le score des deux joueurs est affiché en temps réel en haut de l’écran. Aucun gagnant pour le moment : le jeu est toujours en cours et chaque point compte ! Ce qui rend l’expérience encore plus amusante, c’est que le chat en direct est actif. On peut voir que les joueurs communiquent entre eux pendant qu’ils jouent, ajoutant une dimension sociale et interactive à la partie.
![Capture d'écran 2025-04-14 201507](https://github.com/user-attachments/assets/f67c4d79-25fa-45b1-a4e1-a6c6ce26726a)
 

🏆 Victoire du Joueur 1 !
Le Joueur 1 atteint le score de 10 points et remporte la partie.
Un message s’affiche à l’écran pour annoncer le gagnant.
L’utilisateur a alors le choix :
<.> Appuyer sur R pour revenir au menu.
<.> Appuyer sur Q pour quitter le jeu
![Capture d'écran 2025-04-14 200413](https://github.com/user-attachments/assets/433c5481-953e-4f62-aa0e-fdf5b17add7f)

Game Over. battu par l’IA ! 🤖
Cette photo capture le moment où le joueur a été vaincu par l’agent IA, qui a atteint le score de 10 points. Le match est terminé, et l'écran affiche clairement la défaite. Comme toujours, le joueur a le choix : appuyer sur "Q" pour quitter le jeu, ou sur "R" pour recommencer une nouvelle partie depuis le menu principal. Une défaite… mais peut-être pas la dernière !
![Capture d'écran 2025-04-29 234609](https://github.com/user-attachments/assets/0d25e30f-a8f7-453d-b524-bc4bacf150d7)



