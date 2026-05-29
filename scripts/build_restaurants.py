"""Phase 1a: Consolidation des 2717 JSON en un DataFrame restaurants."""
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scripts.config import RAW_DIR, PROCESSED_DIR


def parse_restaurant(filepath):
    """Extrait les champs clés d'un fichier JSON restaurant."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    r = data[0]
    loc = r.get("location", {}) or {}
    geo = loc.get("geoPoint", {}) or {}
    details = loc.get("details", {}) or {}
    gtm = details.get("gtmData", {}) or {}
    loc_data = gtm.get("locationData", {}) or {}

    # Cuisines
    cuisines = [c["tag"]["localizedName"] for c in
                (r.get("cuisines", {}) or {}).get("items", []) or []]

    # Quartiers
    neighborhoods = [n["neighborhood"]["name"] for n in
                     (r.get("neighborhoods", []) or [])]

    # Prix
    price_items = (r.get("priceTypes", {}) or {}).get("items", []) or []
    price_level = price_items[0]["secondaryName"] if price_items else None

    # Awards Michelin
    awards = r.get("restaurantAwards", []) or []
    has_michelin = any(
        a.get("awards") and a["awards"][0].get("award_title") == "MICHELIN"
        for a in awards
    )

    # Menu
    menu = r.get("menu", {}) or {}
    provider = menu.get("providerContent", {}) or {}
    provider_menu = provider.get("providerMenu", {}) or {}
    menu_list = provider_menu.get("menu") or []

    has_menu = len(menu_list) > 0 and any(
        any(s.get("items") for s in (m.get("sections") or []))
        for m in menu_list
    )

    # Review
    review = r.get("reviewSummary", {}) or {}

    return {
        "locationId": r.get("locationId"),
        "name": r.get("name"),
        "rating": review.get("rating"),
        "reviewCount": review.get("count", 0),
        "latitude": geo.get("latitude"),
        "longitude": geo.get("longitude"),
        "cuisines": "|".join(cuisines) if cuisines else None,
        "neighborhoods": "|".join(neighborhoods) if neighborhoods else None,
        "priceLevel": price_level,
        "hasMichelin": has_michelin,
        "hasMenu": has_menu,
        "city": loc_data.get("cityName"),
        "region": loc_data.get("regionName"),
    }


def main():
    files = sorted(RAW_DIR.glob("*.json"))
    rows = []
    errors = []

    for fp in tqdm(files, desc="Parsing restaurants"):
        try:
            rows.append(parse_restaurant(fp))
        except Exception as e:
            errors.append((fp.name, str(e)))

    df = pd.DataFrame(rows)
    df = df.sort_values("locationId").reset_index(drop=True)

    output = PROCESSED_DIR / "restaurants.parquet"
    df.to_parquet(output, index=False)

    print(f"Restaurants: {len(df)}")
    print(f"With menu:  {df['hasMenu'].sum()}")
    print(f"With rating: {df['rating'].notna().sum()}")
    print(f"Erreurs:     {len(errors)}")
    if errors:
        for name, err in errors[:10]:
            print(f"  {name}: {err}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
