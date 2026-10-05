"""
BYBIT LIVE BOT - SYMBOL SCANNER (Task 14B)
Standalone: queries live Bybit API for cost/tradeability of 12 edges.
Saves results to symbol_costs.json.
"""
import os
import sys
import json
import math
from datetime import datetime
from typing import Dict, List

import config
from bybit_api import BybitAPI
from logger_setup import setup_logging
from dotenv import load_dotenv

try:
    from colorama import Fore, Style, init
    from tabulate import tabulate
    init(autoreset=True)
except ImportError:
    print("Missing: colorama, tabulate. Run: pip install colorama tabulate")
    sys.exit(1)

def main():
    log = setup_logging("INFO")
    print(f"{Fore.CYAN}{Style.BRIGHT}=============================================")
    print(f"{Fore.CYAN}{Style.BRIGHT}  BYBIT SYMBOL SCANNER (Task 14B)")
    print(f"{Fore.CYAN}{Style.BRIGHT}  Live API check of 12 edges")
    print(f"{Fore.CYAN}{Style.BRIGHT}=============================================")

    # Load .env
    env_path = os.path.join(config.BASE_PATH, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    testnet = os.getenv("BYBIT_TESTNET", "false").lower() in ("true", "1", "yes")

    if not api_key or api_key == "your_api_key_here":
        print(f"{Fore.YELLOW}[WARN] No API key set. Running public-only scan.")
    else:
        print(f"[OK] API key loaded, testnet={testnet}")

    api = BybitAPI(api_key, api_secret, testnet=testnet)

    # Fetch account balance if keys available
    balance = None
    if api_key and api_key != "your_api_key_here":
        try:
            balance = api.get_wallet_balance()
            if balance is not None:
                print(f"[OK] Current USDT balance: ${balance:.2f}")
            else:
                print(f"{Fore.YELLOW}[WARN] Could not fetch balance (permissions?)")
        except Exception as e:
            print(f"{Fore.RED}[ERR] Balance fetch failed: {e}")

    equity = balance if balance is not None else config.ACCOUNT_SIZE

    # Scan each edge
    results = []
    for edge in config.ALL_EDGES:
        sym = edge["symbol"]
        print(f"\n  Scanning {sym} ({edge['strategy'][:24]:<24}, TF={edge['tf']:<3}, {edge['priority']})...")
        try:
            info = api.get_instrument_info(sym)
            ticker = api.get_funding_rate(sym)
            if info is None:
                print(f"    {Fore.RED}[ERR] Instrument info unavailable")
                results.append({"edge_id":edge["id"], "symbol":sym, "tradeable":False,
                                 "reason":"Instrument info unavailable"})
                continue
            if ticker is None:
                print(f"    {Fore.YELLOW}[WARN] Ticker unavailable")
                ticker = {"funding_rate":0, "last_price":0, "bid":0, "ask":0}

            min_notional = info["min_notional_value"]
            leverage = min(config.LEVERAGE, int(info["max_leverage"]))
            min_margin = min_notional / leverage
            last_price = ticker.get("last_price", 0)
            bid = ticker.get("bid", 0)
            ask = ticker.get("ask", 0)
            spread_pct = ((ask - bid) / bid * 100) if bid > 0 and ask > 0 else 0
            funding = ticker.get("funding_rate", 0)
            can_afford = equity >= min_margin
            max_concurrent = int(equity / min_margin) if min_margin > 0 else 0

            entry = {
                "edge_id": edge["id"],
                "symbol": sym,
                "strategy": edge["strategy"],
                "tf": edge["tf"],
                "priority": edge["priority"],
                "min_order_qty": info["min_order_qty"],
                "qty_step": info["qty_step"],
                "min_notional_value": min_notional,
                "tick_size": info["tick_size"],
                "max_leverage": info["max_leverage"],
                "chosen_leverage": leverage,
                "min_margin_required": round(min_margin, 4),
                "current_funding_rate": funding,
                "spread_pct": round(spread_pct, 4),
                "last_price": last_price,
                "volume_24h": ticker.get("volume_24h", 0),
                "tradeable": can_afford,
                "max_concurrent_at_current_equity": max_concurrent,
                "reason": "OK" if can_afford else f"Min margin ${min_margin:.2f} > equity ${equity:.2f}",
            }
            results.append(entry)
            status_col = Fore.GREEN if can_afford else Fore.RED
            print(f"    Min notional: ${min_notional:.2f} | Margin needed: ${min_margin:.4f} | {status_col}{'OK' if can_afford else 'SKIP'}")
            print(f"    Funding: {funding*100:.4f}% | Spread: {spread_pct:.3f}%")
        except Exception as e:
            print(f"    {Fore.RED}[ERR] {e}")
            results.append({"edge_id":edge["id"], "symbol":sym, "tradeable":False, "reason":str(e)})

    # Summary table
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=============================================")
    print(f"{Fore.CYAN}{Style.BRIGHT}  SUMMARY - TRADEABILITY WITH ${equity:.2f}")
    print(f"{Fore.CYAN}{Style.BRIGHT}=============================================")

    tbl = []
    for r in results:
        if "min_notional_value" in r:
            col = Fore.GREEN if r["tradeable"] else Fore.RED
            tbl.append([
                r["edge_id"], r["symbol"], r["priority"],
                f"${r['min_notional_value']:.2f}",
                f"${r['min_margin_required']:.4f}",
                f"{r['current_funding_rate']*100:+.4f}%",
                f"{r['spread_pct']:.3f}%",
                f"{col}{'YES' if r['tradeable'] else 'NO'}{Style.RESET_ALL}",
                r["reason"][:30],
            ])
        else:
            tbl.append([r["edge_id"], r["symbol"], "-", "-", "-", "-", "-",
                        f"{Fore.RED}ERR{Style.RESET_ALL}", r.get("reason","")[:30]])
    print(tabulate(tbl, headers=["#","Symbol","Priority","Min Notional","Min Margin","Funding","Spread","Tradeable","Note"],
                   tablefmt="grid"))

    tradeable_count = sum(1 for r in results if r.get("tradeable"))
    print(f"\n  {Fore.GREEN if tradeable_count>=6 else Fore.YELLOW}Tradeable edges: {tradeable_count}/{len(results)}")

    # Save JSON
    out = {
        "scan_time": datetime.utcnow().isoformat(),
        "account_equity_used": equity,
        "leverage": config.LEVERAGE,
        "results": results,
    }
    with open(config.SYMBOL_COSTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  [SAVED] {config.SYMBOL_COSTS_FILE}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"FATAL: {e}")
        import traceback
        traceback.print_exc()
