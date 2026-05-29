"""Phase 1b: Extraction et classification des items de menu."""
import json
import re
import math
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scripts.config import (
    RAW_DIR, PROCESSED_DIR,
    DRINK_MENU_KEYWORDS, STARTER_MENU_KEYWORDS,
    MAIN_MENU_KEYWORDS, DESSERT_MENU_KEYWORDS,
)


def clean_price(text):
    """Convertit un prix texte en float. '5,00€' -> 5.0, '12,50€' -> 12.5"""
    if not text or not isinstance(text, str):
        return None
    cleaned = text.replace("€", "").replace("\xa0", "").replace(" ", "").strip()
    cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def classify_by_menu_title(menu_title):
    """Classifie un menu entier via son titre."""
    if not menu_title:
        return None
    title = menu_title.lower().strip()
    for kw in DRINK_MENU_KEYWORDS:
        if kw in title:
            return "boisson"
    for kw in DESSERT_MENU_KEYWORDS:
        if kw in title:
            return "dessert"
    for kw in STARTER_MENU_KEYWORDS:
        if kw in title:
            return "entrée"
    for kw in MAIN_MENU_KEYWORDS:
        if kw in title:
            return "plat"
    if any(w in title for w in ["menu", "formule", "dégustation", "déjeuner", "dîner"]):
        return "formule"
    return None


def classify_item(item_title, item_desc, menu_category, item_price, all_prices):
    """Classifie un item individuel quand le menu n'a pas de catégorie."""
    if menu_category and menu_category != "formule":
        return menu_category

    text = f"{item_title or ''} {item_desc or ''}".lower().strip()
    if not text:
        return "autre"

    # Boissons (premier filtre car le plus distinctif)
    drink_words = {"café", "cafe", "thé", "the", "chocolat", "infusion",
                   "cappuccino", "espresso", "latte", "macchiato",
                   "coca", "pepsi", "orangina", "schweppes",
                   "evian", "badoit", "san pellegrino", "perrier", "vittel",
                   "eau", "minérale", "minerale", "bouteille", "demi",
                   "bière", "biere", "pression", "heineken", "kronenbourg",
                   "guinness", "ipa", "stella", "cidre", "panaché", "panache",
                   "kir", "pastis", "ricard", "whisky", "vodka", "gin",
                   "rhum", "tequila", "get", "amer", "martini", "campari",
                   "mojito", "margarita", "daïquiri", "spritz", "bellini",
                   "sangria", "irish", "cosmopolitan", "negroni", "manhattan",
                   "bloody mary", "pina colada", "mai tai", "long island",
                   "cocktail", "apéritif", "aperitif", "digestif", "alcool",
                   "limonade", "sirop", "diabolo", "jus d'orange", "jus de",
                   "smoothie", "milkshake", "lait", "chocolat chaud"}
    if any(w in text for w in drink_words):
        return "boisson"

    # Desserts
    dessert_indicators = {"dessert", "glace", "sorbet", "gâteau", "gateau",
                          "tarte", "mousse", "baba", "crème brûlée", "creme brulee",
                          "millefeuille", "mille-feuille", "macaron", "profiterole",
                          "tiramisu", "fondant", "chocolat", "soufflé", "souffle",
                          "île flottante", "ile flottante", "cheesecake", "panna cotta",
                          "éclair", "eclair", "madeleine", "financier", "cannelé",
                          "cannele", "clafoutis", "compote", "beignet", "crêpe suzette",
                          "paris-brest", "tatin", "crumble", "sablé", "sable",
                          "cookie", "brownie", "pain perdu", "brioche",
                          "viennoiserie", "croissant", "salade de fruit", "fruits rouges"}
    if any(w in text for w in dessert_indicators):
        return "dessert"

    # Entrées
    starter_indicators = {"entrée", "entree", "salade", "soupe", "velouté",
                          "veloute", "bisque", "tartare", "carpaccio", "ceviche",
                          "foie gras", "terrine", "rillettes", "œuf", "oeuf",
                          "burrata", "mozzarella", "bruschetta", "crostini",
                          "gaspacho", "gravlax", "saumon fumé", "saumon fume",
                          "huître", "huitre", "escargot", "gambas", "crevette"}
    if any(w in text for w in starter_indicators):
        return "entrée"

    # Plats
    main_indicators = {"entrecôte", "entrecote", "pavé", "pave", "filet",
                       "magret", "confit", "rôti", "roti", "grillé", "grille",
                       "poêlé", "poele", "braisé", "braise", "mijoté", "mijote",
                       "burger", "pizza", "pâtes", "pates", "risotto", "brochette",
                       "côte", "cote", "gigot", "andouillette", "boudin",
                       "cabillaud", "bar", "lotte", "raie", "saint-jacques",
                       "omelette", "galette", "crêpe", "crepe", "wok", "curry",
                       "tournedos", "faux-filet", "bavette", "jarret"}
    if any(w in text for w in main_indicators):
        return "plat"

    # Heuristique prix: si pas classé et prix disponible
    if item_price is not None and len(all_prices) >= 3:
        q1 = all_prices.quantile(0.33)
        q3 = all_prices.quantile(0.67)
        if item_price <= q1:
            return "entrée"
        elif item_price >= q3:
            return "plat"

    return "autre"


def compute_diversity(items_df):
    """Calcule les indicateurs de diversité par restaurant."""
    cats = items_df.groupby("locationId")["category"].value_counts().unstack(fill_value=0)

    for c in ["entrée", "plat", "dessert", "boisson", "formule", "autre"]:
        if c not in cats.columns:
            cats[c] = 0

    cats["nb_items_total"] = cats.sum(axis=1)
    cats["nb_categories"] = (cats[["entrée", "plat", "dessert"]] > 0).sum(axis=1)

    def shannon(row):
        counts = row[["entrée", "plat", "dessert"]].values.astype(float)
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        return float(-sum(p * math.log(p) for p in probs))

    cats["shannon_entropy"] = cats.apply(shannon, axis=1)

    def ttr(grp):
        all_titles = " ".join(grp["item_title"].fillna("")).lower()
        words = [w for w in re.findall(r'\w+', all_titles) if len(w) > 2]
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    ttrs = items_df.groupby("locationId").apply(ttr).rename("type_token_ratio")

    result = cats[["nb_items_total", "nb_categories", "entrée", "plat",
                    "dessert", "boisson", "formule", "autre",
                    "shannon_entropy"]].copy()
    result = result.join(ttrs)
    return result.reset_index()


def main():
    files = sorted(RAW_DIR.glob("*.json"))
    all_items = []
    errors = []

    for fp in tqdm(files, desc="Extracting menus"):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            r = data[0]
            lid = r.get("locationId")
            menu = r.get("menu", {}) or {}
            provider = menu.get("providerContent", {}) or {}
            provider_menu = provider.get("providerMenu", {}) or {}
            menu_list = provider_menu.get("menu") or []

            # Collecter tous les prix du restaurant pour calibration
            all_prices_list = []
            for m in menu_list:
                for s in (m.get("sections") or []):
                    for item in (s.get("items") or []):
                        for p in (item.get("prices") or []):
                            price = clean_price(p.get("priceText_unobsfuscated"))
                            if price is not None:
                                all_prices_list.append(price)

            all_prices = pd.Series(all_prices_list, dtype=float)

            for m in menu_list:
                menu_title = m.get("title_unobsfuscated") or ""
                menu_cat = classify_by_menu_title(menu_title)

                for s in (m.get("sections") or []):
                    section_title = s.get("title_unobsfuscated") or None

                    for item in (s.get("items") or []):
                        item_title = item.get("title_unobsfuscated") or ""
                        item_desc = item.get("description_unobsfuscated") or ""

                        prices = item.get("prices") or []
                        price_text = prices[0].get("priceText_unobsfuscated") if prices else None
                        item_price = clean_price(price_text)

                        category = classify_item(item_title, item_desc,
                                                 menu_cat, item_price, all_prices)

                        all_items.append({
                            "locationId": lid,
                            "menu_title": menu_title,
                            "section_title": section_title,
                            "item_title": item_title,
                            "item_description": item_desc,
                            "item_price": item_price,
                            "category": category,
                        })
        except Exception as e:
            errors.append((fp.name, str(e)))

    items_df = pd.DataFrame(all_items)

    # Sauvegarde table détaillée
    items_output = PROCESSED_DIR / "menu_items.parquet"
    items_df.to_parquet(items_output, index=False)

    # Calcul et sauvegarde diversité
    diversity_df = compute_diversity(items_df)
    div_output = PROCESSED_DIR / "menu_diversity.parquet"
    diversity_df.to_parquet(div_output, index=False)

    print(f"Items extraits:     {len(items_df)}")
    print(f"Restos avec items:  {items_df['locationId'].nunique()}")
    print(f"Categories :")
    for cat, cnt in items_df["category"].value_counts().items():
        print(f"  {cat}: {cnt} ({100*cnt/len(items_df):.1f}%)")
    print(f"Erreurs :             {len(errors)}")
    if errors:
        for name, err in errors[:10]:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
