## Questions pour le Z

1. Pourquoi, dans uart-reader.py, on enlève la denrière colonne du spectrogramme ?
```
            ncol = int(1000 * 10200 / (1e3 * 512)) # ncol = 19
            sgram = sgram[:, :ncol]
```

On obtient donc, après le reshape en une dimension, un spectrogramme de 20*19 = 380.

2. Comment annuler le calcul des spectrogrammes mel sur le mcu et ainsi envoyer uniquement des données audio brutes ?
3. Il faut ensuite considérer la taille des spectrogrammes comme un hyperparamètre. Y a-t-il des mofications à faire dans le main.c ?