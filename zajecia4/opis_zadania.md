### Zakres zadania laboratoryjnego
1. Uruchom dostarczony plik 3_RL_cliffwalking.py i sprawdź, jakie pliki wynikowe są zapisywane.
2. Przeanalizuj strukturę kodu: środowisko, funkcje uczące, ewaluację oraz generowanie wykresów.
3. Wykonaj eksperyment bazowy na domyślnych parametrach i zapisz metryki dla obu algorytmów.
4. Dostrój parametry co najmniej w czterech własnych eksperymentach. Zmieniaj pojedyncze parametry lub
zestawy parametrów w sposób kontrolowany.
5. Porównaj Q-learning i SARSA pod względem szybkości uczenia, stabilności oraz jakości finalnej polityki.
6. Opisz wnioski: kiedy lepszy jest wariant bardziej zachłanny, a kiedy bezpieczniejszy.

### Wymagania do raportu
• Krótki wstęp teoretyczny: czym jest RL i czym różnią się Q-learning oraz SARSA.
• Opis środowiska i sposobu kodowania stanów oraz akcji.
• Tabela eksperymentów z dokładnymi parametrami.
• Co najmniej dwa wykresy i interpretacja zmian w przebiegu uczenia.
• Porównanie finalnych polityk oraz metryk.
• Końcowe wnioski: który algorytm był szybszy, stabilniejszy i bezpieczniejszy.
Tabela do uzupełnienia w raporcie
Eksperyment Algorytm alpha gamma eps_decay Śr.
nagroda
Śr. kroki Wniosek
E1
E2
E3
E4
E5
Pytania pomocnicze do analizy
• Jak wpływa zwiększenie alpha na szybkość i stabilność uczenia?
• Czy wolniejsze zmniejszanie epsilon poprawia jakość finalnej polityki?
• Który algorytm częściej wybiera trasę blisko klifu i dlaczego?
• Czy większa liczba epizodów zawsze poprawia wyniki? Kiedy pojawia się efekt malejących korzyści?
• Jak zmieniają się wyniki po zwiększeniu lub zmniejszeniu kary za klif?


### Co ma znaleźć się w raporcie?
• krótki opis teorii RL oraz różnicy między Q-learning i
SARSA,
•
• opis środowiska i sposobu kodowania stanów i akcji,
•
• tabela eksperymentów z parametrami,
•
• wykresy przebiegu uczenia oraz interpretacja,
•
• porównanie polityk końcowych i metryk,
•
• techniczne wnioski końcowe.
#### Minimum wykonania
Co najmniej 4 własne eksperymenty +
porównanie obu algorytmów na tych samych
ustawieniach.
### Warto ocenić
średnią nagrodę, średnią liczbę kroków,
szybkość stabilizacji wykresu oraz kształt
finalnej polityki