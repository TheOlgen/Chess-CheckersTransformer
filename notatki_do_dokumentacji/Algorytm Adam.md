
(trochę laby 1 ale bardziej rozwinięte)

Działa jak nawigacja, która pomaga modelowi znaleźć najlepsze ustawienia wag (parametrów), minimalizując błędy (stratę). 

Gdy nasz model się uczy to liczy gradient (kierunek, w którym powinien zmienić wagi, by zmniejszyć błąd) i aktualizuje wagi. Zwykły *gradient descent* robi to manualnie,  prowadzi to do :
- Zbyt wolnego uczenia (gdy learning rate jest mały).
- Oscylacji lub rozbieżności (gdy learning rate jest duży).

TODO : lepiej opisać
w skrócie:
- oblicza momentum - śr. gradientów, by nie chodził w kółko 
- Średnia kwadratów gradientów - by dostosować kroki do "trudności" każdego parametru.

objaśnienie - jak Fotyga na numerkach xD
Jak idziemy przez góry i chcemy iść jak najniżej (min. funkcji starty). Zwykły *gradient descent* idzie w dół , ale każdy krok jest tej samej długości. Rezultat - no na logikę nie przejdziesz przez wszystko - przez to że nie myślisz to możesz spaść z jakiejś górki po drodze . Ale jak idziesz z Adamem to on w przeciwieństwie do ciebie myśli i wie by robić mniejsze kroki jak idziesz wyżej. Ogólnie chodzi, że gradienty są hałaśliwe i każda pochodna cząstkowa z danym parametrem jakoś wpływa na nas (prędkość, wysokość itp.). Adam *wygładza* gradient używając wygłuszenia 

- Normalnie: Model widzi pozycję, gdzie ruch `e2e4` raz daje gradient +5.0 (duża wskazówka), a raz +0.1 (mała wskazówka).
- Po wygładzeniu: Adam uśrednia to do np. +2.3 - bardziej stabilnej wartości, która nie rzuca modelem w skrajności.

Model uczy się płynnie, nie dając się zwieść przypadkowym "szarpnięciom" w danych, tak jak doświadczony kierowca ignoruje drobne nierówności na drodze.