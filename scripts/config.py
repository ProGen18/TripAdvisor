"""Configuration centralisée du pipeline TripAdvisor."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "menus_by_location"
PROCESSED_DIR = DATA_DIR / "processed"
ENRICHED_DIR = DATA_DIR / "enriched"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUT_DIR = DATA_DIR / "output"

STATS_DIR = DATA_DIR / "stats"
LEXICONS_DIR = DATA_DIR / "lexicons"

for d in [PROCESSED_DIR, ENRICHED_DIR, EXTERNAL_DIR, OUTPUT_DIR, STATS_DIR, LEXICONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# --- Classification menus ---

DRINK_MENU_KEYWORDS = [
    "boisson", "cocktail", "bière", "bier", "vin", "apéritif", "aperitif",
    "digestif", "café", "cafe", "thé", "the", "chocolat chaud", "drink",
    "soda", "jus", "alcool", "champagne", "spiritueux",
]

STARTER_MENU_KEYWORDS = [
    "entrée", "entree", "starter", "hors d'oeuvre", "hors d'œuvre",
    "amuse", "antipasti", "antipasto",
]

MAIN_MENU_KEYWORDS = [
    "plat", "plats", "main course", "viande", "poisson", "pâtes", "pasta",
    "burger", "pizza", "risotto", "grillade", "rôtisserie", "rotisserie",
    "spécialité", "specialite", "suggestion",
]

DESSERT_MENU_KEYWORDS = [
    "dessert", "fromage", "fromages", "dolci", "sucré", "sucree",
    "pâtisserie", "patisserie", "glace", "glaces", "sorbet",
]

# Lemmes pour classification par item
STARTER_ITEM_LEMMAS = {
    "foie", "gras", "salade", "soupe", "velouté", "veloute", "bisque",
    "tartare", "carpaccio", "ceviche", "raviole", "oeuf", "œuf", "burrata",
    "couteau", "huître", "huitre", "saumon", "fumé", "fume", "terrine",
    "rillettes", "escargot", "gambas", "crevette", "poulpe", "entrée",
    "entree", "crudo", "thon", "mi-cuit",
}

MAIN_ITEM_LEMMAS = {
    "entrecôte", "entrecote", "pavé", "pave", "filet", "magret", "côte",
    "cote", "carré", "carre", "gigot", "confit", "risotto", "pâtes",
    "pates", "burger", "pizza", "wok", "curry", "brochette", "grillé",
    "grille", "rôti", "roti", "poêlé", "poele", "frit", "galette",
    "crêpe", "crepe", "omelette", "dos", "cabillaud", "bar", "lotte",
    "lieu", "raie", "saint-jacques", "homard", "langouste", "tournedos",
    "faux-filet", "bavette", "onglet", "araignée", "araignee",
    "volaille", "poulet", "canard", "agneau", "veau", "porc", "bœuf",
    "boeuf", "jarret", "souris", "pluma", "chorizo", "andouillette",
    "boudin", "cassoulet", "choucroute", "pot-au-feu", "blanquette",
    "daube", "bourguignon",
}

DESSERT_ITEM_LEMMAS = {
    "baba", "tarte", "millefeuille", "mille-feuille", "crème", "creme",
    "brûlée", "brulee", "mousse", "macaron", "profiterole", "tiramisu",
    "glace", "sorbet", "gâteau", "gateau", "fondant", "coulant",
    "chocolat", "citron", "fruit", "rouge", "poire", "pomme",
    "cheesecake", "panna", "cotta", "crêpe", "crepe", "beignet",
    "croustillant", "soufflé", "souffle", "île", "ile", "flottante",
    "paris-brest", "éclair", "eclair", "madeleine", "financier",
    "cannelé", "cannele", "clafoutis", "compote", "salade", "fruit",
    "viennoiserie", "pain", "perdu", "brioche", "crème", "glacée",
}

# Produits de saison (source: RNM, Greenpeace)
SEASONAL_PRODUCTS = {
    "printemps": {
        "asperge", "petit pois", "fève", "feve", "artichaut", "épinard",
        "epinard", "radis", "fraise", "rhubarbe", "morille", "aillet",
        "agneau", "pascal", "navet", "primeur",
    },
    "été": {
        "tomate", "courgette", "aubergine", "poivron", "melon",
        "pastèque", "pasteque", "pêche", "peche", "abricot", "brugnon",
        "nectarine", "figue", "framboise", "myrtille", "groseille",
        "cassis", "mûre", "mure", "salade", "concombre", "haricot",
        "vert", "maïs", "mais", "basilic", "menthe", "coriandre",
    },
    "automne": {
        "champignon", "cèpe", "cepe", "girolle", "trompette", "morille",
        "potiron", "potimarron", "courge", "butternut", "marrons",
        "châtaigne", "chataigne", "raisin", "coing", "poire", "pomme",
        "noix", "noisette", "gibier", "sanglier", "chevreuil", "faisan",
    },
    "hiver": {
        "truffe", "endive", "chou", "poireau", "céleri", "celeri",
        "panais", "rutabaga", "topinambour", "betterave", "mâche",
        "mache", "agrume", "orange", "clémentine", "clemenetine",
        "mandarine", "pamplemousse", "citron", "kaki", "litchi",
        "raclette", "fondue", "tartiflette",
    },
}
