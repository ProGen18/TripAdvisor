"""Tests pour la classification des menus."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import PROJECT_ROOT, PROCESSED_DIR, ENRICHED_DIR, SEASONAL_PRODUCTS
from scripts.classify_menu import clean_price, classify_by_menu_title, classify_item


# --- clean_price ---

def test_clean_price_simple():
    assert clean_price("5,00€") == 5.0
    assert clean_price("12,50€") == 12.5
    assert clean_price("120,00€") == 120.0


def test_clean_price_none():
    assert clean_price(None) is None
    assert clean_price("") is None


def test_clean_price_edge():
    assert clean_price("0,00€") == 0.0
    assert clean_price("9,99€") == 9.99
    assert clean_price("0,50€") == 0.5


def test_clean_price_spaces():
    assert clean_price("  7,00€  ") == 7.0


# --- classify_by_menu_title ---

def test_menu_title_drink():
    assert classify_by_menu_title("Boissons") == "boisson"
    assert classify_by_menu_title("Cocktails & bieres") == "boisson"
    assert classify_by_menu_title("Aperitifs") == "boisson"
    assert classify_by_menu_title("Digestifs") == "boisson"
    assert classify_by_menu_title("Boissons chaudes") == "boisson"


def test_menu_title_dessert():
    assert classify_by_menu_title("Dessert") == "dessert"
    assert classify_by_menu_title("Desserts") == "dessert"
    assert classify_by_menu_title("Nos desserts") == "dessert"
    assert classify_by_menu_title("Les desserts") == "dessert"


def test_menu_title_entree():
    assert classify_by_menu_title("Entree") == "entrée"
    assert classify_by_menu_title("Entrees") == "entrée"
    assert classify_by_menu_title("Les entrees") == "entrée"


def test_menu_title_plat():
    assert classify_by_menu_title("Plat") == "plat"
    assert classify_by_menu_title("Plats") == "plat"
    assert classify_by_menu_title("Plat principal") == "plat"


def test_menu_title_formule():
    assert classify_by_menu_title("Menu Degustation") == "formule"
    assert classify_by_menu_title("Formule midi") == "formule"
    assert classify_by_menu_title("Menu Dejeuner") == "formule"


def test_menu_title_unknown():
    assert classify_by_menu_title("Garniture") is None
    assert classify_by_menu_title("") is None
    assert classify_by_menu_title(None) is None


def test_menu_title_case_insensitive():
    assert classify_by_menu_title("BOISSONS") == "boisson"
    assert classify_by_menu_title("DESSERT") == "dessert"
    assert classify_by_menu_title("ENTREE") == "entrée"


# --- classify_item ---

@pytest.fixture
def empty_prices():
    return pd.Series([], dtype=float)


def test_classify_item_drink(empty_prices):
    assert classify_item("Cafe", "", None, 5.0, empty_prices) == "boisson"
    assert classify_item("Coca-Cola", "", None, 3.0, empty_prices) == "boisson"
    assert classify_item("Mojito", "", None, 12.0, empty_prices) == "boisson"
    assert classify_item("Bouteille d'eau", "", None, 6.0, empty_prices) == "boisson"


def test_classify_item_dessert(empty_prices):
    assert classify_item("Tarte aux pommes", "", None, None, empty_prices) == "dessert"
    assert classify_item("Creme brulee", "", None, None, empty_prices) == "dessert"
    assert classify_item("Tiramisu maison", "", None, None, empty_prices) == "dessert"
    assert classify_item("Millefeuille", "", None, None, empty_prices) == "dessert"


def test_classify_item_entree(empty_prices):
    assert classify_item("Salade Niçoise", "", None, None, empty_prices) == "entrée"
    assert classify_item("Foie gras", "", None, None, empty_prices) == "entrée"
    assert classify_item("Tartare de saumon", "", None, None, empty_prices) == "entrée"
    assert classify_item("Huitres speciales", "", None, None, empty_prices) == "entrée"


def test_classify_item_plat(empty_prices):
    assert classify_item("Entrecote 300g", "", None, None, empty_prices) == "plat"
    assert classify_item("Burger maison", "", None, None, empty_prices) == "plat"
    assert classify_item("Magret de canard", "", None, None, empty_prices) == "plat"
    assert classify_item("Pave de saumon roti", "", None, None, empty_prices) == "plat"


def test_classify_item_inherits_menu_cat(empty_prices):
    """Les items heritent la categorie du menu."""
    assert classify_item("X", "", "dessert", None, empty_prices) == "dessert"
    assert classify_item("Y", "", "entrée", None, empty_prices) == "entrée"
    assert classify_item("Z", "", "plat", None, empty_prices) == "plat"
    assert classify_item("W", "", "boisson", None, empty_prices) == "boisson"


def test_classify_item_formule_not_inherited(empty_prices):
    """Les items dans un menu 'formule' utilisent leur propre classification."""
    # "Salade" est detecte comme entree
    assert classify_item("Salade composee", "", "formule", None, empty_prices) == "entrée"


def test_classify_item_empty_text(empty_prices):
    assert classify_item("", "", None, None, empty_prices) == "autre"


# --- config ---

def test_project_root():
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.name in ("TripAdvisor", "pipeline-restaurants")


def test_directories_created():
    assert PROCESSED_DIR.exists()
    assert ENRICHED_DIR.exists()


def test_seasonal_products_structure():
    assert len(SEASONAL_PRODUCTS) == 4
    for season in ["printemps", "été", "automne", "hiver"]:
        assert season in SEASONAL_PRODUCTS
        assert len(SEASONAL_PRODUCTS[season]) > 5
        assert isinstance(SEASONAL_PRODUCTS[season], set)


def test_seasonal_products_no_overlap_within_season():
    """Verifie qu'il n'y a pas de doublons dans une saison."""
    for season, products in SEASONAL_PRODUCTS.items():
        assert len(products) == len(set(p.strip().lower() for p in products))
