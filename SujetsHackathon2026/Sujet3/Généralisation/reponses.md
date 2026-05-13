PART 1)

1) Les 5 navires avec la fréquence moyenne la plus élevée sont 
0    NAVIRE-9944 Singapore Tanker
1    NAVIRE-9493 China Container
2    NAVIRE-7097 France Fishing
3    NAVIRE-2442 USA Bulker
4    NAVIRE-3265 Denmark Bulker

2) 6 pulse pattern différents existent ('Long-Long-Short',   'Long-Short-Long', 'Short-Short-Short',
  'Short-Short-Long','Continuous','Short-Long-Short')
  et aucun n'apparaît qu'une seule fois dans le dataset

3) Visualition des clusters sur Clustering_signatures_radio.png 
    Avec K=5, nous obtenons logiquement 5 clusters (mal découpés)

PART 3)
Changements brutaux de Fréquence (> 0.5 MHz) 
timestamp  frequency  freq_diff
4004 2026-01-30 00:00:00+00:00     161.92       4.59
1886 2026-02-11 02:00:00+00:00     161.16      -0.76
2856 2026-04-21 04:00:00+00:00     159.77      -1.39
1212 2026-05-10 23:00:00+00:00     160.84       1.07
124  2026-09-26 06:00:00+00:00     159.59      -1.25
4998 2026-10-22 00:00:00+00:00     158.69      -0.90

Moyenne_Frequency  Ecart_Type  Nombre_Signatures
flag                                                              
Denmark                    159.1652      1.7734              478
Bahamas                    159.0598      1.7307              377
Panama                     159.0445      1.7571              640
Marshall Islands           159.0169      1.7181              440
Liberia                    158.9776      1.7927              514
Malta                      158.9753      1.7268              590
USA                        158.9724      1.7450              439
China                      158.9580      1.7374              519
Singapore                  158.9345      1.6932              517
France                     158.8932      1.7485              486

Pavillon avec la fréquence moyenne la plus élevée: Denmark → 159.1652 MHz

Coefficient de corrélation de Pearson : 0.0036
p-value                                : 0.416554
Nombre de paires valides               : 49858

Interprétation : Corrélation très faible positive
❌ La corrélation n'est **pas statistiquement significative** (p >= 0.05)

