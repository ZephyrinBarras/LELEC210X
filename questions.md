## Questions pour le Z

1. Pourquoi, dans uart-reader.py, on enlève la denrière colonne du spectrogramme ?

```
            ncol = int(1000 * 10200 / (1e3 * 512)) # ncol = 19
            sgram = sgram[:, :ncol]
```

On obtient donc, après le reshape en une dimension, un spectrogramme de 20*19 = 380.

2. Comment annuler le calcul des spectrogrammes mel sur le mcu et ainsi envoyer uniquement des données audio brutes ?
3. Il faut ensuite considérer la taille des spectrogrammes comme un hyperparamètre. Y a-t-il des mofications à faire
   dans le main.c ?

## À faire

- Dans python, ajouter du bruit (le code est disponible dans un hands-on) et checker ensuite quelle valeur de tranche de
  temps est la plus avantageuse. C'est-à-dire le nombre de secondes que l'on écoute avant de calculer le spectrogramme.
  Pour chaque valeur de tranche de temps, il y a un nombre d'opérations effectuées. Il faut donc voir l'exactitude du
  modèle pour certaines valeurs de tranche de temps et trouver un compromis : en effet, ce n'est pas forcément une bonne
  idée de doubler le nombre d'opérations jusque pour genre 3% d'exactitude en plus. (les graphes liant tranches de temps
  et nombres d'opérations par seconde a été fait par Zéphyrin en python)
- Aller voir dans spectrogram.c pour retirer l'offset des mesures du micro. Bien voir ce que les assistants font dans le
  spectrogram.c. Zéphyrin a push le code dans la branche classification.