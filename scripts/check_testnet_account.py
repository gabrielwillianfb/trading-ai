"""CLI script to query and display live Binance Spot Testnet account balances and health."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.data.collectors.binance import BinanceClient


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger = logging.getLogger("scripts.check_testnet_account")

    logger.info("=" * 70)
    logger.info("BINANCE SPOT TESTNET ACCOUNT INSPECTOR")
    logger.info(f"Target URL: {settings.EXCHANGE_TESTNET_BASE_URL}")
    logger.info(f"API Key:    {settings.EXCHANGE_API_KEY[:6]}...{settings.EXCHANGE_API_KEY[-4:] if len(settings.EXCHANGE_API_KEY) > 10 else ''}")
    logger.info("=" * 70)

    client = BinanceClient(
        api_key=settings.EXCHANGE_API_KEY,
        api_secret=settings.EXCHANGE_API_SECRET,
        base_url=settings.EXCHANGE_TESTNET_BASE_URL,
    )

    try:
        # 1. Fetch complete raw account info
        account_info = await client.get_account_info()
        can_trade = account_info.get("canTrade", False)
        account_type = account_info.get("accountType", "SPOT")
        update_time = account_info.get("updateTime", 0)

        logger.info(f"Account Type:    {account_type}")
        logger.info(f"Can Trade:       {can_trade}")
        logger.info(f"Last Update:     {update_time}")
        logger.info("-" * 70)
        logger.info("NON-ZERO ASSET BALANCES:")

        balances = account_info.get("balances", [])
        active_balances = [
            b for b in balances if float(b.get("free", 0.0)) > 0 or float(b.get("locked", 0.0)) > 0
        ]

        if not active_balances:
            logger.warning("No non-zero asset balances found in Testnet account.")
        else:
            for b in active_balances:
                asset = b.get("asset")
                free = float(b.get("free", 0.0))
                locked = float(b.get("locked", 0.0))
                total = free + locked
                logger.info(f"  • {asset:<8}: Free = {free:>14.6f} | Locked = {locked:>12.6f} | Total = {total:>14.6f}")

        # 2. Check open orders
        open_orders = await client.get_open_orders()
        logger.info("-" * 70)
        logger.info(f"Active Open Orders on Testnet: {len(open_orders)}")
        for o in open_orders:
            logger.info(f"  • ID {o.get('orderId')} | {o.get('symbol')} {o.get('side')} {o.get('origQty')} @ {o.get('price')} [{o.get('status')}]")

        logger.info("=" * 70)
        logger.info("Testnet Account check completed successfully!")

    except Exception as exc:
        logger.error(f"Failed to query Binance Testnet account: {exc}", exc_info=True)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
