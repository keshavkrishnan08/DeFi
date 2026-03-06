#!/usr/bin/env python3
"""
DeFiLlama Real Data Fetcher
============================
Pulls historical TVL, token prices, and protocol metadata for all 15 DeFi protocols.
Saves everything to CSV files in data/real/ directory.

No API key required. No rate limits.
"""

import urllib.request
import json
import time
import os
import csv
from datetime import datetime, timezone

# ============================================================
# Configuration
# ============================================================

# Protocol slug mapping: our internal name -> DeFiLlama slug
PROTOCOLS = {
    "aave-v3":       {"slug": "aave-v3",       "gecko_id": "aave",            "category": "lending"},
    "aave-v2":       {"slug": "aave-v2",       "gecko_id": "aave",            "category": "lending"},
    "compound-v3":   {"slug": "compound-v3",   "gecko_id": "compound-governance-token", "category": "lending"},
    "compound-v2":   {"slug": "compound-v2",   "gecko_id": "compound-governance-token", "category": "lending"},
    "makerdao":      {"slug": "makerdao",      "gecko_id": "maker",           "category": "cdp"},
    "uniswap-v3":    {"slug": "uniswap-v3",   "gecko_id": "uniswap",         "category": "dex"},
    "uniswap-v2":    {"slug": "uniswap-v2",   "gecko_id": "uniswap",         "category": "dex"},
    "curve-dex":     {"slug": "curve-dex",     "gecko_id": "curve-dao-token", "category": "dex"},
    "lido":          {"slug": "lido",          "gecko_id": "lido-dao",        "category": "staking"},
    "yearn-finance": {"slug": "yearn-finance", "gecko_id": "yearn-finance",   "category": "yield"},
    "convex-finance":{"slug": "convex-finance","gecko_id": "convex-finance",  "category": "yield"},
    "synthetix":     {"slug": "synthetix-v3",  "gecko_id": "havven",          "category": "derivatives"},
    "dydx":          {"slug": "dydx-v4",       "gecko_id": "dydx-chain",      "category": "derivatives"},
    "frax":          {"slug": "frax-ether",    "gecko_id": "frax-share",      "category": "stablecoin"},
    "rocket-pool":   {"slug": "rocket-pool",   "gecko_id": "rocket-pool",     "category": "staking"},
    "morpho":        {"slug": "morpho-v1",     "gecko_id": "morpho",          "category": "lending"},
    "euler":         {"slug": "euler-v2",      "gecko_id": "euler",           "category": "lending"},
    "balancer":      {"slug": "balancer-v2",   "gecko_id": "balancer",        "category": "dex"},
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "real")

# ============================================================
# API Helpers
# ============================================================

def fetch_json(url, retries=3, delay=1):
    """Fetch JSON from URL with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DeFi-Research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Retry {attempt+1}/{retries} for {url[:80]}... ({e})")
                time.sleep(delay * (attempt + 1))
            else:
                print(f"  FAILED: {url[:80]} — {e}")
                return None

# ============================================================
# 1. Fetch TVL History
# ============================================================

def fetch_tvl_history():
    """Fetch daily TVL history for all protocols."""
    print("=" * 60)
    print("FETCHING TVL HISTORY")
    print("=" * 60)

    all_tvl = {}

    for name, info in PROTOCOLS.items():
        slug = info["slug"]
        print(f"  Fetching TVL for {name} (slug: {slug})...", end=" ")

        data = fetch_json(f"https://api.llama.fi/protocol/{slug}")
        if data is None:
            print("FAILED")
            continue

        # Extract total TVL history
        tvl_history = data.get("tvl", [])

        # Also try to get Ethereum-specific TVL
        chain_tvls = data.get("chainTvls", {})
        eth_tvl = chain_tvls.get("Ethereum", {}).get("tvl", [])

        protocol_data = []
        for entry in tvl_history:
            ts = entry.get("date", 0)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            protocol_data.append({
                "date": dt.strftime("%Y-%m-%d"),
                "timestamp": ts,
                "tvl_total": entry.get("totalLiquidityUSD", 0),
            })

        # Merge Ethereum TVL if available
        eth_by_date = {}
        for entry in eth_tvl:
            ts = entry.get("date", 0)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            eth_by_date[dt] = entry.get("totalLiquidityUSD", 0)

        for row in protocol_data:
            row["tvl_ethereum"] = eth_by_date.get(row["date"], 0)

        all_tvl[name] = protocol_data
        print(f"{len(protocol_data)} days ({protocol_data[0]['date'] if protocol_data else 'N/A'} to {protocol_data[-1]['date'] if protocol_data else 'N/A'})")

        time.sleep(0.3)  # polite rate limiting

    return all_tvl

# ============================================================
# 2. Fetch Token Prices
# ============================================================

def fetch_token_prices():
    """Fetch daily token prices using DeFiLlama historical prices endpoint.
    Batches all tokens per day to minimize API calls.
    """
    print("\n" + "=" * 60)
    print("FETCHING TOKEN PRICES")
    print("=" * 60)

    # Build list of unique gecko IDs + ETH/BTC
    gecko_ids = sorted(set(info["gecko_id"] for info in PROTOCOLS.values()))
    gecko_ids_with_macro = gecko_ids + ["ethereum", "bitcoin"]
    coin_ids_str = ",".join(f"coingecko:{gid}" for gid in gecko_ids_with_macro)

    # Generate daily timestamps from 2021-06-01 to today
    from datetime import timedelta
    start_date = datetime(2021, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)
    dates = []
    d = start_date
    while d <= end_date:
        dates.append(d)
        d += timedelta(days=1)

    print(f"  Tokens: {len(gecko_ids_with_macro)}")
    print(f"  Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')} ({len(dates)} days)")
    print(f"  Fetching in batches of 7 days...")

    # Initialize storage
    all_prices = {gid: [] for gid in gecko_ids_with_macro}

    # Fetch every 7 days to reduce API calls (then interpolate if needed)
    step = 7
    fetched = 0
    for i in range(0, len(dates), step):
        d = dates[i]
        ts = int(d.timestamp())
        url = f"https://coins.llama.fi/prices/historical/{ts}/{coin_ids_str}"
        data = fetch_json(url, retries=2, delay=0.5)

        if data and "coins" in data:
            for gid in gecko_ids_with_macro:
                coin_key = f"coingecko:{gid}"
                coin_data = data["coins"].get(coin_key, {})
                if coin_data:
                    all_prices[gid].append({
                        "date": d.strftime("%Y-%m-%d"),
                        "timestamp": ts,
                        "price_usd": coin_data.get("price", 0),
                    })
            fetched += 1

        if fetched % 20 == 0 and fetched > 0:
            print(f"    ...fetched {fetched}/{(len(dates)//step)+1} weeks")
        time.sleep(0.15)  # ~6-7 req/sec

    for gid in gecko_ids_with_macro:
        print(f"  {gid}: {len(all_prices[gid])} price points")

    return all_prices

# ============================================================
# 3. Fetch Yields (Lending Rates)
# ============================================================

def fetch_yields():
    """Fetch yield/APY data from DeFiLlama yields API."""
    print("\n" + "=" * 60)
    print("FETCHING YIELDS (Lending Rates)")
    print("=" * 60)

    data = fetch_json("https://yields.llama.fi/pools")
    if data is None:
        print("  FAILED to fetch yields")
        return {}

    pools = data.get("data", [])
    print(f"  Total pools: {len(pools)}")

    # Filter for our protocols
    protocol_yields = {}
    target_slugs = {info["slug"] for info in PROTOCOLS.values()}

    for pool in pools:
        project = pool.get("project", "")
        if project in target_slugs:
            if project not in protocol_yields:
                protocol_yields[project] = []
            protocol_yields[project].append({
                "pool": pool.get("pool", ""),
                "chain": pool.get("chain", ""),
                "symbol": pool.get("symbol", ""),
                "tvl": pool.get("tvlUsd", 0),
                "apy": pool.get("apy", 0),
                "apy_base": pool.get("apyBase", 0),
                "apy_reward": pool.get("apyReward", 0),
                "il7d": pool.get("il7d"),
                "exposure": pool.get("exposure", ""),
            })

    for slug, pools_list in protocol_yields.items():
        print(f"  {slug}: {len(pools_list)} pools")

    return protocol_yields

# ============================================================
# 4. Fetch Stablecoin Data
# ============================================================

def fetch_stablecoin_data():
    """Fetch stablecoin market cap history."""
    print("\n" + "=" * 60)
    print("FETCHING STABLECOIN DATA")
    print("=" * 60)

    data = fetch_json("https://stablecoins.llama.fi/stablecoins?includePrices=true")
    if data is None:
        return {}

    stables = data.get("peggedAssets", [])
    relevant = {}
    for s in stables:
        name = s.get("name", "").lower()
        if any(x in name for x in ["dai", "usdc", "usdt", "frax", "lusd"]):
            relevant[s["name"]] = {
                "id": s.get("id"),
                "symbol": s.get("symbol"),
                "peg_type": s.get("pegType"),
                "current_mcap": s.get("circulating", {}).get("peggedUSD", 0),
            }
            print(f"  {s['name']}: ${s.get('circulating', {}).get('peggedUSD', 0)/1e9:.2f}B")

    return relevant

# ============================================================
# 5. Fetch Protocol Volume (DEXes)
# ============================================================

def fetch_dex_volumes():
    """Fetch DEX volume data."""
    print("\n" + "=" * 60)
    print("FETCHING DEX VOLUMES")
    print("=" * 60)

    dex_slugs = [name for name, info in PROTOCOLS.items() if info["category"] == "dex"]

    all_volumes = {}
    for name in dex_slugs:
        slug = PROTOCOLS[name]["slug"]
        print(f"  Fetching volume for {name}...", end=" ")
        data = fetch_json(f"https://api.llama.fi/summary/dexs/{slug}?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=true&dataType=dailyVolume")
        if data is None:
            print("FAILED")
            continue

        chart = data.get("totalDataChart", [])
        volumes = []
        for entry in chart:
            if isinstance(entry, list) and len(entry) == 2:
                ts, vol = entry
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                volumes.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "volume_usd": vol,
                })

        all_volumes[name] = volumes
        print(f"{len(volumes)} days")
        time.sleep(0.3)

    return all_volumes

# ============================================================
# 6. Fetch Macro Data (from DeFiLlama ETH price as proxy)
# ============================================================

def fetch_eth_btc_prices():
    """ETH/BTC prices are now fetched in fetch_token_prices(). This is a no-op."""
    print("\n  (ETH/BTC prices included in token price fetch)")
    return {}

# ============================================================
# Save Functions
# ============================================================

def save_tvl(all_tvl):
    """Save TVL data: one CSV per protocol + combined."""
    os.makedirs(os.path.join(OUTPUT_DIR, "tvl"), exist_ok=True)

    # Per-protocol files
    for name, data in all_tvl.items():
        filepath = os.path.join(OUTPUT_DIR, "tvl", f"{name}_tvl.csv")
        if not data:
            continue
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "timestamp", "tvl_total", "tvl_ethereum"])
            writer.writeheader()
            writer.writerows(data)

    # Combined file: rows = dates, columns = protocols
    all_dates = sorted(set(d["date"] for data in all_tvl.values() for d in data))
    combined_path = os.path.join(OUTPUT_DIR, "tvl_combined.csv")
    with open(combined_path, "w", newline="") as f:
        fields = ["date"] + [f"{name}_tvl" for name in all_tvl.keys()]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for date in all_dates:
            row = {"date": date}
            for name, data in all_tvl.items():
                tvl_by_date = {d["date"]: d["tvl_total"] for d in data}
                row[f"{name}_tvl"] = tvl_by_date.get(date, "")
            writer.writerow(row)

    print(f"\n  Saved TVL: {len(all_tvl)} protocols, {len(all_dates)} dates")
    print(f"  -> {combined_path}")

def save_prices(all_prices, protocol_map):
    """Save price data including ETH/BTC macro prices."""
    os.makedirs(os.path.join(OUTPUT_DIR, "prices"), exist_ok=True)

    for gecko_id, data in all_prices.items():
        filepath = os.path.join(OUTPUT_DIR, "prices", f"{gecko_id}_price.csv")
        if not data:
            continue
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "timestamp", "price_usd"])
            writer.writeheader()
            writer.writerows(data)

    # Combined price file (all tokens including ETH/BTC)
    all_dates = sorted(set(d["date"] for data in all_prices.values() for d in data if data))
    combined_path = os.path.join(OUTPUT_DIR, "prices_combined.csv")
    with open(combined_path, "w", newline="") as f:
        fields = ["date"] + [f"{gid}_price" for gid in sorted(all_prices.keys())]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for date in all_dates:
            row = {"date": date}
            for gid, data in all_prices.items():
                price_by_date = {d["date"]: d["price_usd"] for d in data}
                row[f"{gid}_price"] = price_by_date.get(date, "")
            writer.writerow(row)

    print(f"\n  Saved prices: {len(all_prices)} tokens, {len(all_dates)} dates")
    print(f"  -> {combined_path}")

    # Also save ETH/BTC as macro file
    eth_data = {d["date"]: d["price_usd"] for d in all_prices.get("ethereum", [])}
    btc_data = {d["date"]: d["price_usd"] for d in all_prices.get("bitcoin", [])}
    macro_path = os.path.join(OUTPUT_DIR, "macro_prices.csv")
    with open(macro_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "eth_price", "btc_price"])
        writer.writeheader()
        for date in all_dates:
            writer.writerow({
                "date": date,
                "eth_price": eth_data.get(date, ""),
                "btc_price": btc_data.get(date, ""),
            })
    print(f"  Saved macro prices: {len(all_dates)} dates -> {macro_path}")

def save_yields(yields_data):
    """Save yield data."""
    filepath = os.path.join(OUTPUT_DIR, "yields_snapshot.csv")
    rows = []
    for slug, pools in yields_data.items():
        for pool in pools:
            pool["protocol_slug"] = slug
            rows.append(pool)

    if rows:
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Saved yields: {len(rows)} pools -> {filepath}")

def save_volumes(volumes_data):
    """Save DEX volume data."""
    os.makedirs(os.path.join(OUTPUT_DIR, "volumes"), exist_ok=True)

    for name, data in volumes_data.items():
        filepath = os.path.join(OUTPUT_DIR, "volumes", f"{name}_volume.csv")
        if not data:
            continue
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "volume_usd"])
            writer.writeheader()
            writer.writerows(data)

    print(f"\n  Saved volumes: {len(volumes_data)} DEXes")

def save_macro(macro_prices):
    """Save ETH/BTC macro price data."""
    filepath = os.path.join(OUTPUT_DIR, "macro_prices.csv")
    # Merge ETH and BTC by date
    all_dates = sorted(set(
        d["date"] for coin_data in macro_prices.values() for d in coin_data
    ))
    eth_by_date = {d["date"]: d["price"] for d in macro_prices.get("ETH", [])}
    btc_by_date = {d["date"]: d["price"] for d in macro_prices.get("BTC", [])}

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "eth_price", "btc_price"])
        writer.writeheader()
        for date in all_dates:
            writer.writerow({
                "date": date,
                "eth_price": eth_by_date.get(date, ""),
                "btc_price": btc_by_date.get(date, ""),
            })

    print(f"\n  Saved macro: {len(all_dates)} dates -> {filepath}")

def save_metadata():
    """Save protocol metadata."""
    filepath = os.path.join(OUTPUT_DIR, "protocol_metadata.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "slug", "gecko_id", "category"])
        writer.writeheader()
        for name, info in PROTOCOLS.items():
            writer.writerow({"name": name, **info})
    print(f"  Saved metadata -> {filepath}")

# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Protocols: {len(PROTOCOLS)}")
    print()

    # 1. TVL
    all_tvl = fetch_tvl_history()

    # 2. Token Prices
    all_prices = fetch_token_prices()

    # 3. Yields
    yields_data = fetch_yields()

    # 4. Stablecoins
    stable_data = fetch_stablecoin_data()

    # 5. DEX Volumes
    volumes_data = fetch_dex_volumes()

    # 6. ETH/BTC Macro
    macro_prices = fetch_eth_btc_prices()

    # Save everything
    print("\n" + "=" * 60)
    print("SAVING DATA")
    print("=" * 60)

    save_metadata()
    save_tvl(all_tvl)
    save_prices(all_prices, PROTOCOLS)
    save_yields(yields_data)
    save_volumes(volumes_data)

    # Summary
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)

    total_files = 0
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            total_files += 1
            total_size += sz
            print(f"  {os.path.relpath(fp, OUTPUT_DIR):45s} {sz/1024:8.1f} KB")

    print(f"\nTotal: {total_files} files, {total_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
