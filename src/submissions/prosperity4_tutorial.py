import json
from collections import deque

from datamodel import Order, Symbol, TradingState


class Trader:
    def __init__(self) -> None:
        # Best on tutorial backtest data:
        # window=20, buy_offset=2, sell_offset=3, limit=20.
        self.params = {
            "TOMATOES": {
                "window": 20,
                "buy_offset": 2,
                "sell_offset": 3,
                "limit": 20,
                "inventory_skew": 0.08,
                "close_flatten_ts": 195000,
            },
            "EMERALDS": {
                "window": 20,
                "buy_offset": 2,
                "sell_offset": 3,
                "limit": 20,
                "inventory_skew": 0.08,
                "close_flatten_ts": 195000,
            },
        }

    def run(self, state: TradingState):
        data = json.loads(state.traderData) if state.traderData else {}
        orders: dict[Symbol, list[Order]] = {}

        for product in ("TOMATOES", "EMERALDS"):
            symbol = self.resolve_symbol(state, product)
            if symbol is None or symbol not in state.order_depths:
                continue

            sym_data = data.get(product, {})
            product_orders, next_sym_data = self.trade_threshold(
                state=state,
                symbol=symbol,
                product=product,
                sym_data=sym_data,
            )
            data[product] = next_sym_data
            if product_orders:
                orders[symbol] = product_orders

        trader_data = json.dumps(data, separators=(",", ":"))
        conversions = 0
        return orders, conversions, trader_data

    def resolve_symbol(self, state: TradingState, product: str) -> Symbol | None:
        aliases = {
            "TOMATOES": {"tomatoes", "tomato", "tg01"},
            "EMERALDS": {"emeralds", "emerald", "tg02"},
        }
        targets = aliases[product]

        for symbol in state.order_depths.keys():
            if symbol.lower() in targets:
                return symbol

        for symbol, listing in state.listings.items():
            listing_product = listing["product"] if isinstance(listing, dict) else listing.product
            if listing_product.lower() in targets and symbol in state.order_depths:
                return symbol

        return None

    def trade_threshold(
        self,
        state: TradingState,
        symbol: Symbol,
        product: str,
        sym_data: dict,
    ) -> tuple[list[Order], dict]:
        cfg = self.params[product]
        order_depth = state.order_depths[symbol]
        best_bid, best_ask, best_bid_volume, best_ask_volume = self.best_prices(order_depth)

        if best_bid is None or best_ask is None:
            return [], sym_data

        mid = (best_bid + best_ask) / 2.0
        mids = deque(sym_data.get("mids", []), maxlen=cfg["window"])
        mids.append(mid)
        fair = sum(mids) / len(mids)

        position = state.position.get(symbol, 0)
        buy_capacity = cfg["limit"] - position
        sell_capacity = cfg["limit"] + position
        skewed_fair = fair - cfg["inventory_skew"] * position

        orders: list[Order] = []

        # End-of-day risk control: aggressively flatten remaining inventory.
        if state.timestamp >= cfg["close_flatten_ts"]:
            if position > 0 and sell_capacity > 0:
                qty = min(position, sell_capacity, best_bid_volume)
                if qty > 0:
                    orders.append(Order(symbol, best_bid, -qty))
            elif position < 0 and buy_capacity > 0:
                qty = min(-position, buy_capacity, -best_ask_volume)
                if qty > 0:
                    orders.append(Order(symbol, best_ask, qty))

            next_sym_data = {"mids": list(mids)}
            return orders, next_sym_data

        if buy_capacity > 0 and best_ask <= skewed_fair - cfg["buy_offset"]:
            qty = min(buy_capacity, -best_ask_volume)
            if qty > 0:
                orders.append(Order(symbol, best_ask, qty))

        if sell_capacity > 0 and best_bid >= skewed_fair + cfg["sell_offset"]:
            qty = min(sell_capacity, best_bid_volume)
            if qty > 0:
                orders.append(Order(symbol, best_bid, -qty))

        next_sym_data = {"mids": list(mids)}
        return orders, next_sym_data

    def best_prices(self, order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None, None, None, None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_volume = order_depth.buy_orders[best_bid]
        best_ask_volume = order_depth.sell_orders[best_ask]
        return best_bid, best_ask, best_bid_volume, best_ask_volume
