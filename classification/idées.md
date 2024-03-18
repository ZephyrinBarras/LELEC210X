## Idées données par l'assistant (Olivier Leblanc)

- Faire la PCA (ou une autre méthode de réduction de dimension) sur le MCU avant l'envoi. Calculer la PCA sur le MCU
  avant ou après le calcul du spectrogramme. Redemander à Olivier Leblanc s'il faut faire la PCA sur le son son brute ou
  sur le spectrogramme qui a été calculé par le MCU. Le librairie DSP (digital signal processing) en C a été améliorée
  donc il faut aller voir ce côté-là.
- Il est possible, si l'on a envie, d'utiliser un autre modèle que le CNN pour faire la classification. Par exemple, on
  pourrait utiliser un modèle de type RNN (Recurrent Neural Network) ou un modèle de type LSTM (Long Short-Term Memory).
  Il est possible de faire des recherches sur les modèles de classification de sons pour voir si un autre modèle que le
  CNN pourrait être plus adapté à notre problème. C'est seulement si l'on a envie d'apprendre. Si on se chauffe, il faut
  en tout cas que cela apporte une réelle valeur ajoutée à notre rapport sinon ça ne sert un peu à rien. Pytorch est
  sans doute un peu meilleure que Tensorflow car il est maintenu par le communauté nous disait Olivier. Il ne faut pas
  hésiter à aller lui demander de l'aide car il bosse avec cette librairie dans sa recherche.
- Pour effectuer la transformation en mel, nous multiplions simplement les deux matrices (stft et mel_transformation).
  Ce calcul prend beaucoup de temps et pourrait être raccourci en créant une fonction qui ignore les valeurs nulles de
  la matrice de transformation en mel. Pour cela, il faudrait faire une matrice de même taille où un 1 est placé à
  chaque endroit où un coefficient est non-nulle. Ensuite, une petite fonctione calculerait le produit matricielle en
  prenant en compte l'emplacement des valeurs non nulles et nulles de la matrice de transformation. Si on ne passe pas
  par un spectrogramme mel mais par un simple spectrogramme (via par exemple une PCA), la transformation peut être
  ignorée.