import re
import os
import argparse
import pandas as pd

STATE_MAP = {"MICHIGAN": "MI", "MI": "MI", "MICH": "MI", "MICH.": "MI"}

# ------------------------ helpers ------------------------

def as_str(x):
    if pd.isna(x):
        return None
    s = str(x).strip().strip('"').strip()
    return s if s and s != "." else None

def normalize_state(s, default_state=None):
    s = as_str(s)
    if not s and default_state:
        return default_state
    if not s:
        return None
    s_up = s.upper()
    if s_up in STATE_MAP:
        return STATE_MAP[s_up]
    if len(s_up) == 2 and s_up.isalpha():
        return s_up
    return default_state

def clean_zip(z):
    s = as_str(z)
    if s is None:
        return None
    if re.fullmatch(r"\d+(\.0+)?", s):
        try:
            return f"{int(float(s)):05d}"
        except Exception:
            pass
    hits = re.findall(r"\b(\d{5})\b", s)
    return hits[-1] if hits else None

def looks_like_street(s):
    if not s:
        return False
    s2 = s.strip().lower()
    if s2 in {"ap", "apt", "bl"}:
        return False
    return bool(re.search(r"[a-zA-Z]", s)) and bool(re.search(r"\d", s))

def parse_full_address(s):
    out = {"street": None, "city": None, "state": None, "zip": None}
    s = as_str(s)
    if not s:
        return out
    s = re.sub(r"\s+", " ", s)

    # "street, city, state zip"
    m = re.search(
        r"^(?P<street>.+?),\s*(?P<city>[A-Za-z.\s]+?),\s*(?P<state>[A-Za-z]{2,})\s+(?P<zip>\d{5})(?:-\d{4})?$",
        s, flags=re.IGNORECASE)
    if m:
        out.update({k: as_str(v) for k, v in m.groupdict().items()})
        return out

    # "street, city zip"
    m = re.search(
        r"^(?P<street>.+?),\s*(?P<city>[A-Za-z.\s]+?),?\s*(?P<zip>\d{5})(?:-\d{4})?$",
        s, flags=re.IGNORECASE)
    if m:
        out.update({k: as_str(v) for k, v in m.groupdict().items()})
        return out

    # Fallback around last ZIP
    z = clean_zip(s)
    if z:
        parts = s.split(",")
        if parts:
            out["street"] = as_str(parts[0])
            head = s[len(parts[0])+1:].strip() if len(parts) > 1 else ""
            city_match = re.search(
                r"([A-Za-z][A-Za-z.\s]+?)(?:,| +(?:MI|MICHIGAN|[A-Za-z]{2}))?\s+" + z,
                head, flags=re.IGNORECASE)
            if city_match:
                out["city"] = as_str(city_match.group(1))
            out["zip"] = z
    return out

def first_nonempty(*vals, validator=None):
    for v in vals:
        v2 = as_str(v)
        if validator:
            if validator(v2):
                return v2
        else:
            if v2:
                return v2
    return None

def load_table(path, sheet_name=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ".txt"}:
        return pd.read_csv(path, dtype=str)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name, dtype=str)
    raise ValueError(f"Unsupported input extension: {ext}")

def save_table(df, out_path):
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_excel(out_path, index=False)

def find_col(df, target_lc):
    """Find a column by lowercase name without changing original headers."""
    for c in df.columns:
        if c.strip().lower() == target_lc:
            return c
    return None

# ------------------------ main ------------------------

def clean_addresses(in_path, out_path, sheet_name=None, default_state="MI"):
    """
    Keep all original columns; add ParsedStreetAddress/City/State/Zip.
    Drop rows only where ParsedStreetAddress is missing.
    """
    df = load_table(in_path, sheet_name=sheet_name)

    # Case-insensitive lookups for inputs
    col_street_candidates = [x for x in [
        find_col(df, "streetaddress"),
        find_col(df, "addressline1")
    ] if x]
    col_city = find_col(df, "city")
    col_full = find_col(df, "fulladdress")
    col_zip  = find_col(df, "zip")

    # Parse FullAddress once
    if col_full:
        fa_parsed = df[col_full].apply(parse_full_address)
        fa_parsed = pd.DataFrame(list(fa_parsed.values), index=df.index)
    else:
        fa_parsed = pd.DataFrame({"street": None, "city": None, "state": None, "zip": None}, index=df.index)

    street, city, state, zip5 = [], [], [], []
    for i, row in df.iterrows():
        street_val = first_nonempty(
            *[row.get(c) for c in col_street_candidates],
            fa_parsed.at[i, "street"],
            validator=looks_like_street
        )
        city_val = first_nonempty(row.get(col_city) if col_city else None, fa_parsed.at[i, "city"])
        zip_val = first_nonempty(
            clean_zip(row.get(col_zip)) if col_zip else None,
            clean_zip(fa_parsed.at[i, "zip"])
        )
        state_val = normalize_state(
            fa_parsed.at[i, "state"],
            default_state=default_state if (street_val or city_val or zip_val) else None
        )

        street.append(street_val)
        city.append(as_str(city_val))
        state.append(state_val)
        zip5.append(zip_val)

    # ---- append (do not overwrite originals) ----
    df["ParsedStreetAddress"] = street
    df["ParsedCity"]          = [c.title() if isinstance(c, str) else c for c in city]
    df["ParsedState"]         = [s.upper() if isinstance(s, str) else s for s in state]
    df["ParsedZip"]           = zip5

    # drop only when parsed street missing
    df = df[df["ParsedStreetAddress"].notna() & (df["ParsedStreetAddress"].str.strip() != "")].copy()

    save_table(df, out_path)
    return df

# ------------------------ CLI ------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Append parsed address columns; keep originals.")
    p.add_argument("input", help="Path to input (.csv, .xlsx, .xls)")
    p.add_argument("-o", "--output", default="addresses_cleaned.xlsx", help="Output path (.xlsx or .csv)")
    p.add_argument("--sheet", default=None, help="Excel sheet name or index (ignored for CSV)")
    p.add_argument("--default-state", default="MI", help="Fallback state ('' to disable)")
    args = p.parse_args()

    sheet = int(args.sheet) if (args.sheet and args.sheet.isdigit()) else (args.sheet if args.sheet else None)
    default_state = args.default_state if args.default_state.strip() else None

    clean_addresses(args.input, args.output, sheet_name=sheet, default_state=default_state)
