# Veikkausliiga Monte Carlo Simulator

Monte Carlo -simulaattori Veikkausliigan runkosarjan loppusijoitusten arviointiin.

Ohjelman perusidea:

1. Päivitä nykyinen sarjataulukko tiedostoon `data/current_table.csv`.
2. Päivitä jäljellä olevat runkosarjaottelut tiedostoon `data/remaining_fixtures.csv`.
3. Aja simulaatio.
4. Tulokset kirjoitetaan Exceliin `output/simulation_results.xlsx`.

Ohjelma on tehty niin, että voit muuttaa lähtödataa käsin ja ajaa what-if-skenaarioita.

## Asennus

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ajo

Nopea testiajo:

```bash
python run_simulation.py --simulations 10000
```

Varsinainen raskaampi ajo:

```bash
python run_simulation.py --simulations 1000000
```

Oletusasetukset ovat tiedostossa `config.json`.

## Syötetiedostot

### `data/current_table.csv`

Pakolliset sarakkeet:

| Sarake | Selitys |
|---|---|
| `Team` | Joukkueen nimi. Pitää vastata fixture-tiedoston nimiä täsmälleen. |
| `P` | Pelatut ottelut |
| `W` | Voitot |
| `D` | Tasapelit |
| `L` | Tappiot |
| `GF` | Tehdyt maalit |
| `GA` | Päästetyt maalit |
| `GD` | Maaliero. Jos puuttuu, ohjelma laskee tämän. |
| `Pts` | Pisteet |

### `data/remaining_fixtures.csv`

Pakolliset sarakkeet:

| Sarake | Selitys |
|---|---|
| `Date` | Ottelupäivä, vapaamuotoinen tekstinä |
| `Home` | Kotijoukkue |
| `Away` | Vierasjoukkue |

## Malli lyhyesti

Simulaatio käyttää Poisson-pohjaista maalimallia, jossa joukkueille lasketaan:

- hyökkäysvoima
- puolustuksen heikkous
- tasapelitaipumus
- tappioresistanssi
- volatiliteetti

Nykykauden havaintoja shrinkataan kohti liigan keskiarvoa, jotta muutaman pelin otos ei hallitse liikaa. Tätä säädetään `config.json`-tiedoston arvoilla:

- `prior_matches_goals`
- `prior_matches_form`
- `draw_strength`
- `loss_resistance_strength`
- `win_strength`
- `volatility_strength`

Mitä suurempi prior-arvo, sitä konservatiivisempi malli.

## Excel-tuloste

Tulostiedostossa on välilehdet:

| Välilehti | Sisältö |
|---|---|
| `Summary` | Pääyhteenveto: keskipisteet, keskisijoitus, voittotodennäköisyys, top-3, top-6 ja jumboriski |
| `PositionDistribution` | Joukkuekohtainen sijoitusjakauma |
| `TargetPoints` | Pistekohtainen sijoitusanalyysi kaikille joukkuehavainnoille yhdessä |
| `TargetPointsByTeam` | Sama pistekohtainen analyysi erikseen jokaiselle joukkueelle |
| `CurrentTable` | Käytetty lähtötaulukko |
| `Fixtures` | Käytetty otteluohjelma |
| `TeamParameters` | Mallin laskemat joukkueparametrit |
| `FixtureModel` | Ottelukohtaiset lambda-arvot ja 1X2-todennäköisyydet ennen/jälkeen korjauksen |
| `ModelSettings` | Malliasetukset |


## TargetPoints-analyysi

`TargetPoints` näyttää jokaiselle runkosarjan lopulliselle pistemäärälle, kuinka usein kyseinen pistemäärä johtaa eri loppusijoituksiin. Tämä auttaa arvioimaan tavoiterajoja, kuten montako pistettä on yleensä riittävästi top-6 -sijoitukseen. `TargetPoints` yhdistää kaikki joukkueet samaan taulukkoon, kun taas `TargetPointsByTeam` laskee saman analyysin erikseen jokaiselle joukkueelle.

## What-if-esimerkkejä

Interille lisää voitto käsin:

1. Avaa `data/current_table.csv`.
2. Lisää Interin `W`-sarakkeeseen +1, `P`-sarakkeeseen +1 ja `Pts`-sarakkeeseen +3.
3. Päivitä `GF`, `GA` ja `GD` halutun tuloksen mukaan.
4. Poista pelattu ottelu `data/remaining_fixtures.csv`-tiedostosta.
5. Aja simulaatio uudestaan.

Manuaalinen skenaario ilman tilastopäivitystä:

- voit vain muuttaa pisteitä `Pts`-sarakkeessa ja ajaa uudestaan
- tämä on nopea tapa testata esimerkiksi "mitä jos HJK voittaa seuraavat kaksi peliä" -tyyppisiä tilanteita

## Huomio

Tämä ei ole vedonlyöntimarkkinan tasoinen ennustemalli. Se on läpinäkyvä ja muokattava simulaattori. Lopputulos riippuu vahvasti siitä, miten hyvin nykyinen sarjataulukko ja otteluohjelma kuvaavat joukkueiden todellista tasoa.
