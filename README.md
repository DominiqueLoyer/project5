# TREC ASSOCIATION PRESS COMPETITION 88-90


## PLEASE USE THIS CITE KEY IF YOU USE THE CODE OR PART OF THE REPORT/BEAMER
**Citation Key: loyerEvaluationModelesRecherche2025**

##### Abstract
Ce rapport détaille l’implémentation et l’évaluation comparative d’un système de recherche d’information (RI) monolingue sur la collection standard TREC Associated Press (AP) 8890. Nous avons étudié l’influence du prétraitement du texte (stemming Porter vs baseline), de la longueur des requêtes (courtes vs longues) et du modèle de pondération (QLD vs BM25). Le système a été développé en Python avec Pyserini sous Java 21. La meilleure configuration de base identifiée (Baseline, Requêtes Longues, BM25) a atteint un MAP de 0.2205. Le stemming s’est avéré moins performant que la baseline (MAP max de 0.1778). Une technique d’amélioration par pseudo-retour de pertinence, RM3, appliquée sur la meilleure configuration baseline, a permis d’atteindre un MAP final de 0.2948, démontrant une amélioration significative (+34)

### METHODOLOGY

THE FILE TREC_AP88-90_5juin2025.py experiment Porter Stemming and RM3. I tried with lemming first but the ouputs where catastrophic. My hypothesis: The lemmer was either to agressive or the Associated Press corpus (243 000 XLM files compressed with TAR) was already pretty cleaned.

I will put the code of the lemmer eventually.

Dominique S. Loyer

![alt text](http://url/to/img.png)
![alt text](http://url/to/img.png)
