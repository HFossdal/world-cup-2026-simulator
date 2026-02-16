"""
World Cup 2026 Simulator — Data Layer

All 48 teams, 12 groups, bracket structure, historical head-to-head,
and team strength ratings based on FIFA rankings and recent form.
"""

import copy

# ---------------------------------------------------------------------------
# Team database
# ---------------------------------------------------------------------------
# attack / defense are Poisson‑model multipliers (baseline = 1.0).
# Higher attack → more goals scored.  Higher defense → fewer goals conceded.
# midfield influences possession simulation.
# form is the average of last‑10‑match results (1 = W, 0.5 = D, 0 = L).
# ---------------------------------------------------------------------------

TEAMS: dict[str, dict] = {
    # ── Group A ──────────────────────────────────────────────────────────
    "MEX": {
        "name": "Mexico", "code": "MEX", "flag": "🇲🇽",
        "confederation": "CONCACAF", "fifa_ranking": 14,
        "attack": 1.55, "defense": 1.50, "midfield": 1.50,
        "form": 0.65,
        "key_players": {
            "FW": ["S. Giménez", "H. Lozano", "R. Jiménez"],
            "MF": ["E. Álvarez", "L. Romo", "O. Pineda"],
            "DF": ["J. Vásquez", "C. Montes", "J. Sánchez"],
            "GK": ["G. Ochoa"],
        },
    },
    "KOR": {
        "name": "South Korea", "code": "KOR", "flag": "🇰🇷",
        "confederation": "AFC", "fifa_ranking": 24,
        "attack": 1.40, "defense": 1.45, "midfield": 1.45,
        "form": 0.60,
        "key_players": {
            "FW": ["Son Heung‑min", "Hwang Hee‑chan", "Oh Hyeon‑gyu"],
            "MF": ["Lee Kang‑in", "Hwang In‑beom", "Jeong Woo‑yeong"],
            "DF": ["Kim Min‑jae", "Kim Jin‑su"],
            "GK": ["Kim Seung‑gyu"],
        },
    },
    "RSA": {
        "name": "South Africa", "code": "RSA", "flag": "🇿🇦",
        "confederation": "CAF", "fifa_ranking": 43,
        "attack": 1.15, "defense": 1.15, "midfield": 1.10,
        "form": 0.45,
        "key_players": {
            "FW": ["P. Tau", "L. Mokoena"],
            "MF": ["T. Mokoena", "S. Mkhize"],
            "DF": ["M. Mudau", "G. Xulu"],
            "GK": ["R. Williams"],
        },
    },
    "DEN": {
        "name": "Denmark", "code": "DEN", "flag": "🇩🇰",
        "confederation": "UEFA", "fifa_ranking": 19,
        "attack": 1.50, "defense": 1.55, "midfield": 1.50,
        "form": 0.60,
        "key_players": {
            "FW": ["R. Højlund", "J. Wind", "A. Olsen"],
            "MF": ["C. Eriksen", "P. Højbjerg", "M. Hjulmand"],
            "DF": ["A. Christensen", "J. Andersen", "V. Kristensen"],
            "GK": ["K. Schmeichel"],
        },
    },
    # ── Group B ──────────────────────────────────────────────────────────
    "CAN": {
        "name": "Canada", "code": "CAN", "flag": "🇨🇦",
        "confederation": "CONCACAF", "fifa_ranking": 35,
        "attack": 1.35, "defense": 1.30, "midfield": 1.30,
        "form": 0.55,
        "key_players": {
            "FW": ["J. David", "A. Davies", "C. Larin"],
            "MF": ["S. Eustáquio", "I. Koné", "T. Buchanan"],
            "DF": ["A. Johnston", "K. Miller", "D. Bombito"],
            "GK": ["M. Crépeau"],
        },
    },
    "SUI": {
        "name": "Switzerland", "code": "SUI", "flag": "🇨🇭",
        "confederation": "UEFA", "fifa_ranking": 17,
        "attack": 1.45, "defense": 1.60, "midfield": 1.50,
        "form": 0.60,
        "key_players": {
            "FW": ["B. Embolo", "N. Okafor", "Z. Amdouni"],
            "MF": ["G. Xhaka", "D. Freuler", "R. Vargas"],
            "DF": ["M. Akanji", "N. Elvedi", "R. Rodríguez"],
            "GK": ["Y. Sommer"],
        },
    },
    "QAT": {
        "name": "Qatar", "code": "QAT", "flag": "🇶🇦",
        "confederation": "AFC", "fifa_ranking": 34,
        "attack": 1.20, "defense": 1.25, "midfield": 1.20,
        "form": 0.45,
        "key_players": {
            "FW": ["Almoez Ali", "Akram Afif", "Mohammed Muntari"],
            "MF": ["Hassan Al‑Haydos", "Abdulaziz Hatem"],
            "DF": ["Bassam Al‑Rawi", "Homam Ahmed"],
            "GK": ["Saad Al‑Sheeb"],
        },
    },
    "ITA": {
        "name": "Italy", "code": "ITA", "flag": "🇮🇹",
        "confederation": "UEFA", "fifa_ranking": 10,
        "attack": 1.55, "defense": 1.80, "midfield": 1.65,
        "form": 0.65,
        "key_players": {
            "FW": ["G. Scamacca", "M. Retegui", "F. Chiesa"],
            "MF": ["N. Barella", "S. Tonali", "L. Pellegrini"],
            "DF": ["A. Bastoni", "R. Calafiori", "G. Di Lorenzo"],
            "GK": ["G. Donnarumma"],
        },
    },
    # ── Group C ──────────────────────────────────────────────────────────
    "BRA": {
        "name": "Brazil", "code": "BRA", "flag": "🇧🇷",
        "confederation": "CONMEBOL", "fifa_ranking": 5,
        "attack": 1.85, "defense": 1.70, "midfield": 1.75,
        "form": 0.65,
        "key_players": {
            "FW": ["Vinícius Jr", "Rodrygo", "Endrick"],
            "MF": ["Bruno Guimarães", "Lucas Paquetá", "Raphinha"],
            "DF": ["Marquinhos", "É. Militão", "G. Magalhães"],
            "GK": ["Alisson"],
        },
    },
    "MAR": {
        "name": "Morocco", "code": "MAR", "flag": "🇲🇦",
        "confederation": "CAF", "fifa_ranking": 16,
        "attack": 1.50, "defense": 1.60, "midfield": 1.55,
        "form": 0.70,
        "key_players": {
            "FW": ["Y. En‑Nesyri", "H. Ziyech", "S. Boufal"],
            "MF": ["S. Amrabat", "A. Ounahi", "I. Bennacer"],
            "DF": ["A. Hakimi", "N. Mazraoui", "N. Aguerd"],
            "GK": ["Y. Bounou"],
        },
    },
    "HAI": {
        "name": "Haiti", "code": "HAI", "flag": "🇭🇹",
        "confederation": "CONCACAF", "fifa_ranking": 47,
        "attack": 0.85, "defense": 0.90, "midfield": 0.85,
        "form": 0.35,
        "key_players": {
            "FW": ["F. Pierrot", "D. Jean‑Jacques"],
            "MF": ["M. Duval", "D. Étienne Jr"],
            "DF": ["C. Hérold", "A. Gedeon"],
            "GK": ["J. Placide"],
        },
    },
    "SCO": {
        "name": "Scotland", "code": "SCO", "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
        "confederation": "UEFA", "fifa_ranking": 27,
        "attack": 1.35, "defense": 1.40, "midfield": 1.40,
        "form": 0.55,
        "key_players": {
            "FW": ["C. Adams", "L. Dykes", "R. Shankland"],
            "MF": ["S. McTominay", "J. McGinn", "B. Gilmour"],
            "DF": ["A. Robertson", "K. Tierney", "J. Hendry"],
            "GK": ["A. Gunn"],
        },
    },
    # ── Group D ──────────────────────────────────────────────────────────
    "USA": {
        "name": "United States", "code": "USA", "flag": "🇺🇸",
        "confederation": "CONCACAF", "fifa_ranking": 15,
        "attack": 1.50, "defense": 1.50, "midfield": 1.50,
        "form": 0.65,
        "key_players": {
            "FW": ["C. Pulisic", "T. Weah", "F. Balogun"],
            "MF": ["W. McKennie", "T. Adams", "G. Reyna"],
            "DF": ["S. Dest", "C. Richards", "T. Robinson"],
            "GK": ["M. Turner"],
        },
    },
    "PAR": {
        "name": "Paraguay", "code": "PAR", "flag": "🇵🇾",
        "confederation": "CONMEBOL", "fifa_ranking": 33,
        "attack": 1.30, "defense": 1.35, "midfield": 1.30,
        "form": 0.50,
        "key_players": {
            "FW": ["A. Enciso", "J. Almirón", "A. Sanabria"],
            "MF": ["M. Villasanti", "A. Cubas", "Ó. Romero"],
            "DF": ["G. Gómez", "F. Balbuena", "O. Alderete"],
            "GK": ["R. Fernández"],
        },
    },
    "AUS": {
        "name": "Australia", "code": "AUS", "flag": "🇦🇺",
        "confederation": "AFC", "fifa_ranking": 28,
        "attack": 1.35, "defense": 1.35, "midfield": 1.30,
        "form": 0.55,
        "key_players": {
            "FW": ["M. Duke", "J. Maclaren", "C. Goodwin"],
            "MF": ["R. McGree", "A. Hrustic", "C. Devlin"],
            "DF": ["H. Souttar", "K. Rowles", "A. Behich"],
            "GK": ["M. Ryan"],
        },
    },
    "TUR": {
        "name": "Türkiye", "code": "TUR", "flag": "🇹🇷",
        "confederation": "UEFA", "fifa_ranking": 20,
        "attack": 1.50, "defense": 1.45, "midfield": 1.45,
        "form": 0.60,
        "key_players": {
            "FW": ["K. Aktürkoğlu", "B. Yılmaz", "Y. Akgün"],
            "MF": ["A. Güler", "H. Çalhanoğlu", "İ. Kahveci"],
            "DF": ["M. Demiral", "F. Kadıoğlu", "S. Özkacar"],
            "GK": ["A. Bayındır"],
        },
    },
    # ── Group E ──────────────────────────────────────────────────────────
    "GER": {
        "name": "Germany", "code": "GER", "flag": "🇩🇪",
        "confederation": "UEFA", "fifa_ranking": 9,
        "attack": 1.75, "defense": 1.70, "midfield": 1.75,
        "form": 0.65,
        "key_players": {
            "FW": ["K. Havertz", "N. Füllkrug", "L. Sané"],
            "MF": ["J. Musiala", "F. Wirtz", "İ. Gündoğan"],
            "DF": ["A. Rüdiger", "J. Tah", "D. Raum"],
            "GK": ["M. ter Stegen"],
        },
    },
    "CUR": {
        "name": "Curaçao", "code": "CUR", "flag": "🇨🇼",
        "confederation": "CONCACAF", "fifa_ranking": 46,
        "attack": 0.90, "defense": 0.95, "midfield": 0.90,
        "form": 0.40,
        "key_players": {
            "FW": ["K. Bacuna", "R. Hooi"],
            "MF": ["L. Bacuna", "J. Rijsdijk"],
            "DF": ["C. Martina", "J. St. Jago"],
            "GK": ["E. Room"],
        },
    },
    "CIV": {
        "name": "Ivory Coast", "code": "CIV", "flag": "🇨🇮",
        "confederation": "CAF", "fifa_ranking": 38,
        "attack": 1.35, "defense": 1.30, "midfield": 1.30,
        "form": 0.55,
        "key_players": {
            "FW": ["S. Haller", "N. Pépé", "W. Zaha"],
            "MF": ["F. Kessié", "I. Sangaré", "S. Fofana"],
            "DF": ["S. Aurier", "E. Bailly", "W. Boly"],
            "GK": ["B. Sangaré"],
        },
    },
    "ECU": {
        "name": "Ecuador", "code": "ECU", "flag": "🇪🇨",
        "confederation": "CONMEBOL", "fifa_ranking": 22,
        "attack": 1.50, "defense": 1.40, "midfield": 1.40,
        "form": 0.55,
        "key_players": {
            "FW": ["E. Valencia", "M. Caicedo", "K. Páez"],
            "MF": ["M. Caicedo", "A. Franco", "J. Cifuentes"],
            "DF": ["P. Hincapié", "F. Torres", "R. Arboleda"],
            "GK": ["H. Galíndez"],
        },
    },
    # ── Group F ──────────────────────────────────────────────────────────
    "NED": {
        "name": "Netherlands", "code": "NED", "flag": "🇳🇱",
        "confederation": "UEFA", "fifa_ranking": 7,
        "attack": 1.75, "defense": 1.70, "midfield": 1.70,
        "form": 0.65,
        "key_players": {
            "FW": ["C. Gakpo", "B. Brobbey", "D. Malen"],
            "MF": ["F. de Jong", "R. Gravenberch", "T. Reijnders"],
            "DF": ["V. van Dijk", "N. Aké", "J. Timber"],
            "GK": ["B. Verbruggen"],
        },
    },
    "JPN": {
        "name": "Japan", "code": "JPN", "flag": "🇯🇵",
        "confederation": "AFC", "fifa_ranking": 18,
        "attack": 1.55, "defense": 1.50, "midfield": 1.55,
        "form": 0.70,
        "key_players": {
            "FW": ["T. Kubo", "K. Mitoma", "A. Doan"],
            "MF": ["W. Endo", "H. Doan", "D. Kamada"],
            "DF": ["K. Itakura", "T. Tomiyasu", "M. Yoshida"],
            "GK": ["S. Suzuki"],
        },
    },
    "POL": {
        "name": "Poland", "code": "POL", "flag": "🇵🇱",
        "confederation": "UEFA", "fifa_ranking": 25,
        "attack": 1.40, "defense": 1.45, "midfield": 1.40,
        "form": 0.55,
        "key_players": {
            "FW": ["R. Lewandowski", "A. Milik", "K. Świderski"],
            "MF": ["P. Zieliński", "S. Szymański", "J. Moder"],
            "DF": ["J. Kiwior", "J. Bednarek", "M. Cash"],
            "GK": ["W. Szczęsny"],
        },
    },
    "TUN": {
        "name": "Tunisia", "code": "TUN", "flag": "🇹🇳",
        "confederation": "CAF", "fifa_ranking": 31,
        "attack": 1.25, "defense": 1.40, "midfield": 1.30,
        "form": 0.50,
        "key_players": {
            "FW": ["A. Talbi", "Y. Msakni", "S. Jaziri"],
            "MF": ["A. Mejbri", "A. Laidouni", "E. Skhiri"],
            "DF": ["D. Bronn", "M. Talbi", "A. Abdi"],
            "GK": ["A. Dahmen"],
        },
    },
    # ── Group G ──────────────────────────────────────────────────────────
    "BEL": {
        "name": "Belgium", "code": "BEL", "flag": "🇧🇪",
        "confederation": "UEFA", "fifa_ranking": 6,
        "attack": 1.70, "defense": 1.65, "midfield": 1.70,
        "form": 0.60,
        "key_players": {
            "FW": ["R. Lukaku", "J. Doku", "L. Openda"],
            "MF": ["K. De Bruyne", "Y. Tielemans", "A. Onana"],
            "DF": ["T. Meunier", "A. Theate", "W. Faes"],
            "GK": ["K. Casteels"],
        },
    },
    "EGY": {
        "name": "Egypt", "code": "EGY", "flag": "🇪🇬",
        "confederation": "CAF", "fifa_ranking": 30,
        "attack": 1.40, "defense": 1.35, "midfield": 1.35,
        "form": 0.55,
        "key_players": {
            "FW": ["M. Salah", "M. Trezeguet", "O. Marmoush"],
            "MF": ["M. Elneny", "E. Ashour", "I. Adel"],
            "DF": ["A. Hegazi", "M. Abdelmonem", "O. Kamal"],
            "GK": ["M. El Shenawy"],
        },
    },
    "IRN": {
        "name": "Iran", "code": "IRN", "flag": "🇮🇷",
        "confederation": "AFC", "fifa_ranking": 26,
        "attack": 1.35, "defense": 1.45, "midfield": 1.35,
        "form": 0.55,
        "key_players": {
            "FW": ["M. Taremi", "S. Azmoun", "K. Ansarifard"],
            "MF": ["S. Ezatolahi", "A. Nourollahi", "A. Jahanbakhsh"],
            "DF": ["S. Hosseini", "M. Pouraliganji", "E. Hajsafi"],
            "GK": ["A. Beiranvand"],
        },
    },
    "NZL": {
        "name": "New Zealand", "code": "NZL", "flag": "🇳🇿",
        "confederation": "OFC", "fifa_ranking": 44,
        "attack": 1.05, "defense": 1.15, "midfield": 1.05,
        "form": 0.40,
        "key_players": {
            "FW": ["C. Wood", "M. Rojas"],
            "MF": ["M. Stamenic", "J. Bell"],
            "DF": ["T. Smith", "N. Reid"],
            "GK": ["S. Marinovic"],
        },
    },
    # ── Group H ──────────────────────────────────────────────────────────
    "ESP": {
        "name": "Spain", "code": "ESP", "flag": "🇪🇸",
        "confederation": "UEFA", "fifa_ranking": 3,
        "attack": 1.85, "defense": 1.80, "midfield": 1.90,
        "form": 0.75,
        "key_players": {
            "FW": ["L. Yamal", "N. Williams", "Á. Morata"],
            "MF": ["Pedri", "Gavi", "Rodri"],
            "DF": ["D. Carvajal", "A. Laporte", "R. Le Normand"],
            "GK": ["Unai Simón"],
        },
    },
    "URU": {
        "name": "Uruguay", "code": "URU", "flag": "🇺🇾",
        "confederation": "CONMEBOL", "fifa_ranking": 13,
        "attack": 1.65, "defense": 1.70, "midfield": 1.55,
        "form": 0.65,
        "key_players": {
            "FW": ["D. Núñez", "F. Pellistri", "M. Araújo"],
            "MF": ["F. Valverde", "R. Bentancur", "M. Vecino"],
            "DF": ["R. Araújo", "J. Giménez", "M. Olivera"],
            "GK": ["S. Rochet"],
        },
    },
    "KSA": {
        "name": "Saudi Arabia", "code": "KSA", "flag": "🇸🇦",
        "confederation": "AFC", "fifa_ranking": 35,
        "attack": 1.25, "defense": 1.30, "midfield": 1.25,
        "form": 0.50,
        "key_players": {
            "FW": ["S. Al‑Dawsari", "F. Al‑Buraikan"],
            "MF": ["S. Al‑Shehri", "M. Kanno", "A. Al‑Malki"],
            "DF": ["A. Al‑Amri", "Y. Al‑Shahrani", "A. Al‑Bulayhi"],
            "GK": ["M. Al‑Owais"],
        },
    },
    "CPV": {
        "name": "Cape Verde", "code": "CPV", "flag": "🇨🇻",
        "confederation": "CAF", "fifa_ranking": 45,
        "attack": 1.05, "defense": 1.10, "midfield": 1.05,
        "form": 0.40,
        "key_players": {
            "FW": ["G. Rodrigues", "L. Biai"],
            "MF": ["K. Borges", "J. Graça"],
            "DF": ["S. Lopes", "R. Fortes"],
            "GK": ["V. Soares"],
        },
    },
    # ── Group I ──────────────────────────────────────────────────────────
    "FRA": {
        "name": "France", "code": "FRA", "flag": "🇫🇷",
        "confederation": "UEFA", "fifa_ranking": 2,
        "attack": 1.90, "defense": 1.85, "midfield": 1.85,
        "form": 0.75,
        "key_players": {
            "FW": ["K. Mbappé", "O. Dembélé", "M. Thuram"],
            "MF": ["A. Tchouaméni", "E. Camavinga", "A. Griezmann"],
            "DF": ["W. Saliba", "D. Upamecano", "T. Hernández"],
            "GK": ["M. Maignan"],
        },
    },
    "SEN": {
        "name": "Senegal", "code": "SEN", "flag": "🇸🇳",
        "confederation": "CAF", "fifa_ranking": 23,
        "attack": 1.45, "defense": 1.45, "midfield": 1.40,
        "form": 0.55,
        "key_players": {
            "FW": ["S. Mané", "I. Sarr", "B. Dia"],
            "MF": ["I. Gueye", "N. Mendy", "P. Gueye"],
            "DF": ["K. Koulibaly", "A. Diallo", "Y. Sabaly"],
            "GK": ["É. Mendy"],
        },
    },
    "NOR": {
        "name": "Norway", "code": "NOR", "flag": "🇳🇴",
        "confederation": "UEFA", "fifa_ranking": 29,
        "attack": 1.55, "defense": 1.30, "midfield": 1.35,
        "form": 0.55,
        "key_players": {
            "FW": ["E. Haaland", "A. Sørloth", "O. Ødegaard"],
            "MF": ["M. Ødegaard", "S. Berge", "F. Aursnes"],
            "DF": ["K. Ajer", "L. Ostigård", "B. Meling"],
            "GK": ["Ø. Nyland"],
        },
    },
    "IRQ": {
        "name": "Iraq", "code": "IRQ", "flag": "🇮🇶",
        "confederation": "AFC", "fifa_ranking": 42,
        "attack": 1.15, "defense": 1.20, "midfield": 1.15,
        "form": 0.45,
        "key_players": {
            "FW": ["A. Bayesh", "M. Ali"],
            "MF": ["I. Bayesh", "A. Fadhel"],
            "DF": ["A. Natiq", "R. Ghanim"],
            "GK": ["J. Hameed"],
        },
    },
    # ── Group J ──────────────────────────────────────────────────────────
    "ARG": {
        "name": "Argentina", "code": "ARG", "flag": "🇦🇷",
        "confederation": "CONMEBOL", "fifa_ranking": 1,
        "attack": 1.95, "defense": 1.85, "midfield": 1.90,
        "form": 0.80,
        "key_players": {
            "FW": ["J. Álvarez", "L. Martínez", "A. Garnacho"],
            "MF": ["E. Fernández", "A. Mac Allister", "R. De Paul"],
            "DF": ["C. Romero", "L. Martínez", "N. Molina"],
            "GK": ["E. Martínez"],
        },
    },
    "ALG": {
        "name": "Algeria", "code": "ALG", "flag": "🇩🇿",
        "confederation": "CAF", "fifa_ranking": 32,
        "attack": 1.30, "defense": 1.35, "midfield": 1.30,
        "form": 0.50,
        "key_players": {
            "FW": ["I. Bennacer", "S. Mahrez", "A. Slimani"],
            "MF": ["I. Bennacer", "S. Atal", "H. Aouar"],
            "DF": ["R. Bensebaini", "A. Mandi", "D. Benlamri"],
            "GK": ["R. M'Bolhi"],
        },
    },
    "AUT": {
        "name": "Austria", "code": "AUT", "flag": "🇦🇹",
        "confederation": "UEFA", "fifa_ranking": 21,
        "attack": 1.45, "defense": 1.45, "midfield": 1.45,
        "form": 0.60,
        "key_players": {
            "FW": ["M. Arnautović", "M. Gregoritsch", "C. Baumgartner"],
            "MF": ["K. Laimer", "F. Grillitsch", "N. Seiwald"],
            "DF": ["D. Alaba", "P. Lienhart", "M. Wöber"],
            "GK": ["P. Pentz"],
        },
    },
    "JOR": {
        "name": "Jordan", "code": "JOR", "flag": "🇯🇴",
        "confederation": "AFC", "fifa_ranking": 40,
        "attack": 1.10, "defense": 1.25, "midfield": 1.15,
        "form": 0.45,
        "key_players": {
            "FW": ["H. Al‑Tamari", "Y. Al‑Rawashdeh"],
            "MF": ["M. Abu Hasan", "N. Al‑Rawabdeh"],
            "DF": ["A. Al‑Naimat", "Y. Al‑Bitar"],
            "GK": ["Y. Shboul"],
        },
    },
    # ── Group K ──────────────────────────────────────────────────────────
    "POR": {
        "name": "Portugal", "code": "POR", "flag": "🇵🇹",
        "confederation": "UEFA", "fifa_ranking": 8,
        "attack": 1.80, "defense": 1.75, "midfield": 1.75,
        "form": 0.70,
        "key_players": {
            "FW": ["R. Leão", "G. Ramos", "P. Félix"],
            "MF": ["B. Fernandes", "B. Silva", "V. Vitinha"],
            "DF": ["R. Dias", "A. Silva", "N. Mendes"],
            "GK": ["D. Costa"],
        },
    },
    "COL": {
        "name": "Colombia", "code": "COL", "flag": "🇨🇴",
        "confederation": "CONMEBOL", "fifa_ranking": 12,
        "attack": 1.65, "defense": 1.55, "midfield": 1.60,
        "form": 0.65,
        "key_players": {
            "FW": ["L. Díaz", "R. Falcao", "J. Córdoba"],
            "MF": ["J. Arias", "J. Cuadrado", "R. Ríos"],
            "DF": ["D. Sánchez", "J. Lucumí", "S. Arias"],
            "GK": ["C. Vargas"],
        },
    },
    "UZB": {
        "name": "Uzbekistan", "code": "UZB", "flag": "🇺🇿",
        "confederation": "AFC", "fifa_ranking": 41,
        "attack": 1.20, "defense": 1.20, "midfield": 1.20,
        "form": 0.45,
        "key_players": {
            "FW": ["E. Shomurodov", "I. Nasimov"],
            "MF": ["J. Khodjaev", "O. Amonov"],
            "DF": ["R. Gafurzhanow", "A. Tuhtasinov"],
            "GK": ["U. Kutlimuratov"],
        },
    },
    "COD": {
        "name": "DR Congo", "code": "COD", "flag": "🇨🇩",
        "confederation": "CAF", "fifa_ranking": 39,
        "attack": 1.25, "defense": 1.20, "midfield": 1.20,
        "form": 0.45,
        "key_players": {
            "FW": ["C. Bakambu", "S. Malango"],
            "MF": ["Y. Wissa", "N. Mbemba"],
            "DF": ["C. Luyindama", "A. Mbemba"],
            "GK": ["J. Kiassumbua"],
        },
    },
    # ── Group L ──────────────────────────────────────────────────────────
    "ENG": {
        "name": "England", "code": "ENG", "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
        "confederation": "UEFA", "fifa_ranking": 4,
        "attack": 1.80, "defense": 1.75, "midfield": 1.80,
        "form": 0.70,
        "key_players": {
            "FW": ["B. Saka", "P. Foden", "H. Kane"],
            "MF": ["J. Bellingham", "D. Rice", "C. Palmer"],
            "DF": ["J. Stones", "M. Guéhi", "T. Alexander‑Arnold"],
            "GK": ["J. Pickford"],
        },
    },
    "CRO": {
        "name": "Croatia", "code": "CRO", "flag": "🇭🇷",
        "confederation": "UEFA", "fifa_ranking": 11,
        "attack": 1.60, "defense": 1.70, "midfield": 1.70,
        "form": 0.60,
        "key_players": {
            "FW": ["A. Kramarić", "B. Petković", "A. Budimir"],
            "MF": ["L. Modrić", "M. Brozović", "M. Kovačić"],
            "DF": ["J. Gvardiol", "D. Šutalo", "J. Stanišić"],
            "GK": ["D. Livaković"],
        },
    },
    "GHA": {
        "name": "Ghana", "code": "GHA", "flag": "🇬🇭",
        "confederation": "CAF", "fifa_ranking": 36,
        "attack": 1.30, "defense": 1.25, "midfield": 1.25,
        "form": 0.45,
        "key_players": {
            "FW": ["I. Williams", "J. Ayew", "A. Kudus"],
            "MF": ["M. Kudus", "T. Partey", "E. Sulemana"],
            "DF": ["A. Djiku", "D. Amartey", "T. Lamptey"],
            "GK": ["L. Jojo Wollacott"],
        },
    },
    "PAN": {
        "name": "Panama", "code": "PAN", "flag": "🇵🇦",
        "confederation": "CONCACAF", "fifa_ranking": 37,
        "attack": 1.15, "defense": 1.30, "midfield": 1.20,
        "form": 0.45,
        "key_players": {
            "FW": ["J. Fajardo", "E. Guerrero"],
            "MF": ["A. Carrasquilla", "É. Davis"],
            "DF": ["F. Escobar", "H. Cummings"],
            "GK": ["O. Mosquera"],
        },
    },
    # ── Playoff candidate teams (not yet qualified) ───────────────
    "NIR": {
        "name": "Northern Ireland", "code": "NIR", "flag": "🇬🇧",
        "confederation": "UEFA", "fifa_ranking": 52,
        "attack": 1.00, "defense": 1.10, "midfield": 1.05,
        "form": 0.40,
        "key_players": {
            "FW": ["C. Washington", "D. Charles"],
            "MF": ["S. Davis", "A. McCann"],
            "DF": ["J. Evans", "P. McNair"],
            "GK": ["B. Peacock-Farrell"],
        },
    },
    "WAL": {
        "name": "Wales", "code": "WAL", "flag": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
        "confederation": "UEFA", "fifa_ranking": 48,
        "attack": 1.10, "defense": 1.15, "midfield": 1.10,
        "form": 0.40,
        "key_players": {
            "FW": ["B. Johnson", "M. Harris"],
            "MF": ["A. Ramsey", "H. Wilson"],
            "DF": ["B. Davies", "C. Mepham"],
            "GK": ["D. Ward"],
        },
    },
    "BIH": {
        "name": "Bosnia & Herzegovina", "code": "BIH", "flag": "🇧🇦",
        "confederation": "UEFA", "fifa_ranking": 50,
        "attack": 1.15, "defense": 1.10, "midfield": 1.10,
        "form": 0.40,
        "key_players": {
            "FW": ["E. Džeko", "E. Demirović"],
            "MF": ["M. Pjanić", "A. Hajradinović"],
            "DF": ["S. Kolašinac", "E. Bičakčić"],
            "GK": ["N. Vasilj"],
        },
    },
    "UKR": {
        "name": "Ukraine", "code": "UKR", "flag": "🇺🇦",
        "confederation": "UEFA", "fifa_ranking": 22,
        "attack": 1.40, "defense": 1.40, "midfield": 1.40,
        "form": 0.55,
        "key_players": {
            "FW": ["A. Dovbyk", "M. Mudryk"],
            "MF": ["O. Zinchenko", "R. Malinovskyi"],
            "DF": ["I. Zabarnyi", "M. Matviyenko"],
            "GK": ["A. Lunin"],
        },
    },
    "SWE": {
        "name": "Sweden", "code": "SWE", "flag": "🇸🇪",
        "confederation": "UEFA", "fifa_ranking": 23,
        "attack": 1.35, "defense": 1.40, "midfield": 1.35,
        "form": 0.50,
        "key_players": {
            "FW": ["A. Isak", "V. Gyökeres"],
            "MF": ["D. Kulusevski", "J. Svanberg"],
            "DF": ["V. Lindelöf", "E. Krafth"],
            "GK": ["R. Olsen"],
        },
    },
    "ALB": {
        "name": "Albania", "code": "ALB", "flag": "🇦🇱",
        "confederation": "UEFA", "fifa_ranking": 46,
        "attack": 1.10, "defense": 1.20, "midfield": 1.10,
        "form": 0.45,
        "key_players": {
            "FW": ["A. Broja", "J. Bajrami"],
            "MF": ["K. Asllani", "N. Bajrami"],
            "DF": ["B. Djimsiti", "E. Hysaj"],
            "GK": ["T. Strakosha"],
        },
    },
    "ROU": {
        "name": "Romania", "code": "ROU", "flag": "🇷🇴",
        "confederation": "UEFA", "fifa_ranking": 30,
        "attack": 1.25, "defense": 1.30, "midfield": 1.25,
        "form": 0.50,
        "key_players": {
            "FW": ["D. Drăgușin", "G. Pușcaș"],
            "MF": ["N. Stanciu", "R. Marin"],
            "DF": ["A. Rațiu", "R. Drăgușin"],
            "GK": ["F. Niță"],
        },
    },
    "SVK": {
        "name": "Slovakia", "code": "SVK", "flag": "🇸🇰",
        "confederation": "UEFA", "fifa_ranking": 38,
        "attack": 1.15, "defense": 1.25, "midfield": 1.15,
        "form": 0.45,
        "key_players": {
            "FW": ["R. Boženík", "D. Strelec"],
            "MF": ["S. Lobotka", "O. Duda"],
            "DF": ["M. Škriniar", "D. Hancko"],
            "GK": ["M. Dúbravka"],
        },
    },
    "XKX": {
        "name": "Kosovo", "code": "XKX", "flag": "🇽🇰",
        "confederation": "UEFA", "fifa_ranking": 55,
        "attack": 1.05, "defense": 1.05, "midfield": 1.05,
        "form": 0.40,
        "key_players": {
            "FW": ["V. Muriqi", "E. Rashica"],
            "MF": ["M. Vojvoda", "A. Zhegrova"],
            "DF": ["A. Rrahmani", "F. Hadergjonaj"],
            "GK": ["A. Muric"],
        },
    },
    "MKD": {
        "name": "North Macedonia", "code": "MKD", "flag": "🇲🇰",
        "confederation": "UEFA", "fifa_ranking": 56,
        "attack": 1.05, "defense": 1.05, "midfield": 1.00,
        "form": 0.35,
        "key_players": {
            "FW": ["A. Trajkovski", "B. Miovski"],
            "MF": ["E. Elmas", "E. Bardhi"],
            "DF": ["S. Ristovski", "D. Velkoski"],
            "GK": ["S. Dimitrievski"],
        },
    },
    "CZE": {
        "name": "Czech Republic", "code": "CZE", "flag": "🇨🇿",
        "confederation": "UEFA", "fifa_ranking": 35,
        "attack": 1.20, "defense": 1.25, "midfield": 1.20,
        "form": 0.45,
        "key_players": {
            "FW": ["P. Schick", "A. Hložek"],
            "MF": ["A. Barák", "T. Souček"],
            "DF": ["T. Holeš", "L. Krejčí"],
            "GK": ["J. Staněk"],
        },
    },
    "IRL": {
        "name": "Republic of Ireland", "code": "IRL", "flag": "🇮🇪",
        "confederation": "UEFA", "fifa_ranking": 51,
        "attack": 1.00, "defense": 1.15, "midfield": 1.05,
        "form": 0.35,
        "key_players": {
            "FW": ["E. Ferguson", "T. Idah"],
            "MF": ["J. Knight", "J. Cullen"],
            "DF": ["N. Collins", "S. Coleman"],
            "GK": ["C. Kelleher"],
        },
    },
    "NCL": {
        "name": "New Caledonia", "code": "NCL", "flag": "🇳🇨",
        "confederation": "OFC", "fifa_ranking": 80,
        "attack": 0.70, "defense": 0.75, "midfield": 0.70,
        "form": 0.30,
        "key_players": {
            "FW": ["G. Gope-Fenepej", "J. Hmae"],
            "MF": ["B. Kaicone", "R. Cama"],
            "DF": ["J. Wahnawe", "D. Xalite"],
            "GK": ["R. Deloy"],
        },
    },
    "JAM": {
        "name": "Jamaica", "code": "JAM", "flag": "🇯🇲",
        "confederation": "CONCACAF", "fifa_ranking": 54,
        "attack": 1.05, "defense": 1.05, "midfield": 1.00,
        "form": 0.35,
        "key_players": {
            "FW": ["M. Antonio", "L. Nicholson"],
            "MF": ["L. Bailey", "B. Reid"],
            "DF": ["A. Blake", "D. Lowe"],
            "GK": ["A. Blake"],
        },
    },
    "BOL": {
        "name": "Bolivia", "code": "BOL", "flag": "🇧🇴",
        "confederation": "CONMEBOL", "fifa_ranking": 60,
        "attack": 0.95, "defense": 1.00, "midfield": 0.95,
        "form": 0.35,
        "key_players": {
            "FW": ["R. Vaca", "C. Suárez"],
            "MF": ["M. Villarroel", "F. Saucedo"],
            "DF": ["J. Sagredo", "A. Jusino"],
            "GK": ["C. Lampe"],
        },
    },
    "SUR": {
        "name": "Suriname", "code": "SUR", "flag": "🇸🇷",
        "confederation": "CONCACAF", "fifa_ranking": 65,
        "attack": 0.90, "defense": 0.90, "midfield": 0.85,
        "form": 0.30,
        "key_players": {
            "FW": ["G. Haps", "I. Becker"],
            "MF": ["R. Donk", "S. Yeini"],
            "DF": ["K. Leerdam", "D. Pinas"],
            "GK": ["W. Hato"],
        },
    },
}

# ---------------------------------------------------------------------------
# Group assignments  (official FIFA World Cup 2026 draw, Dec 5 2025)
# ---------------------------------------------------------------------------
# Playoff placeholders: ITA (UEFA PO A), POL (UEFA PO B), TUR (UEFA PO C),
# DEN (UEFA PO D), COD (IC PO 1), IRQ (IC PO 2).
# ---------------------------------------------------------------------------

GROUPS: dict[str, list[str]] = {
    "A": ["MEX", "KOR", "RSA", "DEN"],
    "B": ["CAN", "SUI", "QAT", "ITA"],
    "C": ["BRA", "MAR", "HAI", "SCO"],
    "D": ["USA", "PAR", "AUS", "TUR"],
    "E": ["GER", "CUR", "CIV", "ECU"],
    "F": ["NED", "JPN", "POL", "TUN"],
    "G": ["BEL", "EGY", "IRN", "NZL"],
    "H": ["ESP", "URU", "KSA", "CPV"],
    "I": ["FRA", "SEN", "NOR", "IRQ"],
    "J": ["ARG", "ALG", "AUT", "JOR"],
    "K": ["POR", "COL", "UZB", "COD"],
    "L": ["ENG", "CRO", "GHA", "PAN"],
}

# ---------------------------------------------------------------------------
# Knockout bracket  – Round of 32 structure
# ---------------------------------------------------------------------------
# Each entry: (team_slot_A, team_slot_B)
# "1X" = winner of group X,  "2X" = runner-up of group X
# "3_..." = best third-place team from one of the listed groups
# ---------------------------------------------------------------------------

ROUND_OF_32: list[dict] = [
    # Match 73
    {"id": 73, "slot_a": "2A", "slot_b": "2B",
     "label": "2A vs 2B"},
    # Match 74
    {"id": 74, "slot_a": "1E", "slot_b": "3_ABCDF",
     "label": "1E vs 3rd (A/B/C/D/F)"},
    # Match 75
    {"id": 75, "slot_a": "1F", "slot_b": "2C",
     "label": "1F vs 2C"},
    # Match 76
    {"id": 76, "slot_a": "1C", "slot_b": "2F",
     "label": "1C vs 2F"},
    # Match 77
    {"id": 77, "slot_a": "1I", "slot_b": "3_CDFGH",
     "label": "1I vs 3rd (C/D/F/G/H)"},
    # Match 78
    {"id": 78, "slot_a": "2E", "slot_b": "2I",
     "label": "2E vs 2I"},
    # Match 79
    {"id": 79, "slot_a": "1A", "slot_b": "3_CEFHI",
     "label": "1A vs 3rd (C/E/F/H/I)"},
    # Match 80
    {"id": 80, "slot_a": "1L", "slot_b": "3_EHIJK",
     "label": "1L vs 3rd (E/H/I/J/K)"},
    # Match 81
    {"id": 81, "slot_a": "1D", "slot_b": "3_BEFIJ",
     "label": "1D vs 3rd (B/E/F/I/J)"},
    # Match 82
    {"id": 82, "slot_a": "1G", "slot_b": "3_AEHIJ",
     "label": "1G vs 3rd (A/E/H/I/J)"},
    # Match 83
    {"id": 83, "slot_a": "2K", "slot_b": "2L",
     "label": "2K vs 2L"},
    # Match 84
    {"id": 84, "slot_a": "1H", "slot_b": "2J",
     "label": "1H vs 2J"},
    # Match 85
    {"id": 85, "slot_a": "1B", "slot_b": "3_EFGIJ",
     "label": "1B vs 3rd (E/F/G/I/J)"},
    # Match 86
    {"id": 86, "slot_a": "1J", "slot_b": "2H",
     "label": "1J vs 2H"},
    # Match 87
    {"id": 87, "slot_a": "1K", "slot_b": "3_DEIJL",
     "label": "1K vs 3rd (D/E/I/J/L)"},
    # Match 88
    {"id": 88, "slot_a": "2D", "slot_b": "2G",
     "label": "2D vs 2G"},
]

# Round of 16 pairings — indices into ROUND_OF_32 list (0‑15)
ROUND_OF_16_FEEDS: list[tuple[int, int]] = [
    (0, 1),    # R16‑1: W(M73) vs W(M74)
    (2, 3),    # R16‑2: W(M75) vs W(M76)
    (4, 5),    # R16‑3: W(M77) vs W(M78)
    (6, 7),    # R16‑4: W(M79) vs W(M80)
    (8, 9),    # R16‑5: W(M81) vs W(M82)
    (10, 11),  # R16‑6: W(M83) vs W(M84)
    (12, 13),  # R16‑7: W(M85) vs W(M86)
    (14, 15),  # R16‑8: W(M87) vs W(M88)
]

# QF pairings (indices into ROUND_OF_16_FEEDS winners)
QF_FEEDS: list[tuple[int, int]] = [
    (0, 1),  # QF1: W(R16-1) vs W(R16-2)
    (2, 3),  # QF2: W(R16-3) vs W(R16-4)
    (4, 5),  # QF3: W(R16-5) vs W(R16-6)
    (6, 7),  # QF4: W(R16-7) vs W(R16-8)
]

# SF pairings (indices into QF winners)
SF_FEEDS: list[tuple[int, int]] = [
    (0, 1),  # SF1: W(QF1) vs W(QF2)
    (2, 3),  # SF2: W(QF3) vs W(QF4)
]

# ---------------------------------------------------------------------------
# Third-place team bracket assignment
# ---------------------------------------------------------------------------
# Each R32 slot that expects a 3rd-place team lists which groups are valid.
# The assignment function uses backtracking to find a valid mapping.
# ---------------------------------------------------------------------------

THIRD_PLACE_SLOTS: dict[int, set[str]] = {
    74: set("ABCDF"),
    77: set("CDFGH"),
    79: set("CEFHI"),
    80: set("EHIJK"),
    81: set("BEFIJ"),
    82: set("AEHIJ"),
    85: set("EFGIJ"),
    87: set("DEIJL"),
}


def assign_third_place_teams(qualifying_groups: list[str]) -> dict[int, str]:
    """Given the 8 group letters whose 3rd-place teams qualify, return
    a mapping of R32 match id → group letter using backtracking."""
    slots = sorted(THIRD_PLACE_SLOTS.keys())
    assignment: dict[int, str] = {}
    remaining = set(qualifying_groups)

    def backtrack(idx: int) -> bool:
        if idx == len(slots):
            return len(remaining) == 0
        slot = slots[idx]
        for group in sorted(THIRD_PLACE_SLOTS[slot] & remaining):
            assignment[slot] = group
            remaining.discard(group)
            if backtrack(idx + 1):
                return True
            remaining.add(group)
            del assignment[slot]
        return False

    if backtrack(0):
        return assignment
    # Fallback: assign greedily if backtracking fails
    remaining = list(qualifying_groups)
    result: dict[int, str] = {}
    for slot in slots:
        for g in remaining:
            if g in THIRD_PLACE_SLOTS[slot]:
                result[slot] = g
                remaining.remove(g)
                break
    return result


# ---------------------------------------------------------------------------
# Historical head-to-head data (major matchups)
# ---------------------------------------------------------------------------
# Format: { (team_a, team_b): {"played": N, "a_wins": W, "draws": D, "b_wins": W} }
# ---------------------------------------------------------------------------

HEAD_TO_HEAD: dict[tuple[str, str], dict] = {
    ("ARG", "BRA"): {"played": 111, "a_wins": 40, "draws": 26, "b_wins": 45},
    ("ARG", "FRA"): {"played": 14, "a_wins": 6, "draws": 3, "b_wins": 5},
    ("ARG", "GER"): {"played": 25, "a_wins": 7, "draws": 8, "b_wins": 10},
    ("ARG", "ENG"): {"played": 16, "a_wins": 7, "draws": 3, "b_wins": 6},
    ("ARG", "URU"): {"played": 197, "a_wins": 89, "draws": 44, "b_wins": 64},
    ("BRA", "GER"): {"played": 25, "a_wins": 13, "draws": 5, "b_wins": 7},
    ("BRA", "FRA"): {"played": 15, "a_wins": 5, "draws": 3, "b_wins": 7},
    ("BRA", "ENG"): {"played": 27, "a_wins": 12, "draws": 6, "b_wins": 9},
    ("BRA", "ESP"): {"played": 11, "a_wins": 3, "draws": 4, "b_wins": 4},
    ("BRA", "NED"): {"played": 13, "a_wins": 5, "draws": 3, "b_wins": 5},
    ("ENG", "GER"): {"played": 37, "a_wins": 13, "draws": 9, "b_wins": 15},
    ("ENG", "FRA"): {"played": 33, "a_wins": 17, "draws": 7, "b_wins": 9},
    ("ENG", "ESP"): {"played": 27, "a_wins": 7, "draws": 8, "b_wins": 12},
    ("ENG", "CRO"): {"played": 11, "a_wins": 5, "draws": 3, "b_wins": 3},
    ("FRA", "GER"): {"played": 34, "a_wins": 11, "draws": 8, "b_wins": 15},
    ("FRA", "ESP"): {"played": 36, "a_wins": 12, "draws": 8, "b_wins": 16},
    ("FRA", "POR"): {"played": 28, "a_wins": 10, "draws": 8, "b_wins": 10},
    ("FRA", "BEL"): {"played": 75, "a_wins": 30, "draws": 19, "b_wins": 26},
    ("GER", "NED"): {"played": 46, "a_wins": 16, "draws": 12, "b_wins": 18},
    ("GER", "ESP"): {"played": 26, "a_wins": 9, "draws": 7, "b_wins": 10},
    ("GER", "ITA"): {"played": 37, "a_wins": 9, "draws": 12, "b_wins": 16},
    ("ESP", "POR"): {"played": 41, "a_wins": 16, "draws": 11, "b_wins": 14},
    ("ESP", "NED"): {"played": 16, "a_wins": 7, "draws": 3, "b_wins": 6},
    ("ESP", "ITA"): {"played": 39, "a_wins": 12, "draws": 14, "b_wins": 13},
    ("NED", "BEL"): {"played": 128, "a_wins": 56, "draws": 24, "b_wins": 48},
    ("POR", "NED"): {"played": 14, "a_wins": 5, "draws": 3, "b_wins": 6},
    ("MEX", "USA"): {"played": 77, "a_wins": 36, "draws": 15, "b_wins": 26},
    ("KOR", "JPN"): {"played": 85, "a_wins": 29, "draws": 23, "b_wins": 33},
    ("URU", "PAR"): {"played": 98, "a_wins": 42, "draws": 22, "b_wins": 34},
    ("COL", "BRA"): {"played": 37, "a_wins": 5, "draws": 10, "b_wins": 22},
    ("ENG", "SCO"): {"played": 115, "a_wins": 48, "draws": 25, "b_wins": 42},
    ("NOR", "FRA"): {"played": 17, "a_wins": 3, "draws": 5, "b_wins": 9},
}


def get_h2h(team_a: str, team_b: str) -> dict | None:
    """Return H2H record between two teams (order-independent)."""
    key = (team_a, team_b)
    if key in HEAD_TO_HEAD:
        return HEAD_TO_HEAD[key]
    rev = (team_b, team_a)
    if rev in HEAD_TO_HEAD:
        r = HEAD_TO_HEAD[rev]
        return {
            "played": r["played"],
            "a_wins": r["b_wins"],
            "draws": r["draws"],
            "b_wins": r["a_wins"],
        }
    return None


# ---------------------------------------------------------------------------
# Qualification status — 42 confirmed, 6 undecided (March 2026 playoffs)
# ---------------------------------------------------------------------------

CONFIRMED_QUALIFIED: set[str] = {
    # AFC (8)
    "AUS", "IRN", "JPN", "JOR", "QAT", "KSA", "KOR", "UZB",
    # CAF (9)
    "ALG", "CPV", "EGY", "GHA", "CIV", "MAR", "SEN", "RSA", "TUN",
    # UEFA (12)
    "AUT", "BEL", "CRO", "ENG", "FRA", "GER", "NED", "NOR", "POR", "SCO", "ESP", "SUI",
    # CONCACAF (6)
    "CAN", "CUR", "HAI", "MEX", "PAN", "USA",
    # CONMEBOL (6)
    "ARG", "BRA", "COL", "ECU", "PAR", "URU",
    # OFC (1)
    "NZL",
}

PLAYOFF_SLOTS: list[dict] = [
    {
        "id": "slot_A3",
        "group": "A",
        "position": 3,
        "label": "UEFA Path D",
        "candidates": ["DEN", "MKD", "CZE", "IRL"],
        "most_likely": "DEN",
    },
    {
        "id": "slot_B3",
        "group": "B",
        "position": 3,
        "label": "UEFA Path A",
        "candidates": ["ITA", "NIR", "WAL", "BIH"],
        "most_likely": "ITA",
    },
    {
        "id": "slot_D3",
        "group": "D",
        "position": 3,
        "label": "UEFA Path C",
        "candidates": ["TUR", "ROU", "SVK", "XKX"],
        "most_likely": "TUR",
    },
    {
        "id": "slot_F2",
        "group": "F",
        "position": 1,  # 0-indexed position 1 = 2nd in group
        "label": "UEFA Path B",
        "candidates": ["POL", "UKR", "SWE", "ALB"],
        "most_likely": "POL",
    },
    {
        "id": "slot_I3",
        "group": "I",
        "position": 3,
        "label": "Intercontinental 2",
        "candidates": ["IRQ", "BOL", "SUR"],
        "most_likely": "IRQ",
    },
    {
        "id": "slot_K3",
        "group": "K",
        "position": 3,
        "label": "Intercontinental 1",
        "candidates": ["COD", "JAM", "NCL"],
        "most_likely": "COD",
    },
]

# Map from slot placeholder to default team (most_likely) for quick lookup
_SLOT_DEFAULTS: dict[str, str] = {
    slot["id"]: slot["most_likely"] for slot in PLAYOFF_SLOTS
}

# Map from slot id to (group, position_index) for group replacement
_SLOT_GROUP_POS: dict[str, tuple[str, int]] = {
    "slot_A3": ("A", 3),
    "slot_B3": ("B", 3),
    "slot_D3": ("D", 3),
    "slot_F2": ("F", 2),
    "slot_I3": ("I", 3),
    "slot_K3": ("K", 3),
}

# ---------------------------------------------------------------------------
# Helper: deep-copy mutable team data so scenarios don't corrupt baseline
# ---------------------------------------------------------------------------

def get_teams_copy(selections: dict[str, str] | None = None) -> dict[str, dict]:
    """Return a deep copy of the 48 teams for the tournament.

    *selections* maps playoff slot id → chosen team code.
    If None or incomplete, missing slots use the most_likely defaults.
    Only the 42 confirmed teams plus the 6 selected (or default) teams
    are included.
    """
    selections = selections or {}
    chosen_codes: set[str] = set(CONFIRMED_QUALIFIED)
    for slot in PLAYOFF_SLOTS:
        code = selections.get(slot["id"], slot["most_likely"])
        chosen_codes.add(code)
    return {
        code: copy.deepcopy(TEAMS[code])
        for code in chosen_codes
        if code in TEAMS
    }


def get_groups_copy(selections: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Return a copy of GROUPS with playoff placeholder positions
    swapped to the selected (or default) teams."""
    selections = selections or {}
    groups = copy.deepcopy(GROUPS)
    for slot in PLAYOFF_SLOTS:
        code = selections.get(slot["id"], slot["most_likely"])
        group_letter, pos_idx = _SLOT_GROUP_POS[slot["id"]]
        if pos_idx < len(groups[group_letter]):
            groups[group_letter][pos_idx] = code
    return groups
