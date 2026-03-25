import json
from datamodel import Order, OrderDepth, Symbol, TradingState


class Trader:
    def __init__(self) -> None:
        self.limits: dict[str, int] = {
            "EMERALD": 20,
            "TOMATO": 20,
        }

    def run(self, state: TradingState):
        trader_data = json.loads(state.traderData) if state.traderData else {}
        orders: dict[Symbol, list[Order]] = {}

        emerald_symbol = self.resolve_symbol(state, ["EMERALDS", "EMERALD", "TG02"])
        tomato_symbol = self.resolve_symbol(state, ["TOMATOES", "TOMATO", "TG01"])

        if emerald_symbol and emerald_symbol in state.order_depths:
            emerald_orders = self.trade_emerald(state, emerald_symbol)
            if emerald_orders:
                orders[emerald_symbol] = emerald_orders

        if tomato_symbol and tomato_symbol in state.order_depths:
            tomato_orders, tomato_ema, tomato_prev_mid = self.trade_tomato(
                state,
                tomato_symbol,
                trader_data.get("tomato_ema"),
                trader_data.get("tomato_prev_mid"),
            )
            trader_data["tomato_ema"] = tomato_ema
            trader_data["tomato_prev_mid"] = tomato_prev_mid
            if tomato_orders:
                orders[tomato_symbol] = tomato_orders

        conversions = 0
        next_trader_data = json.dumps(trader_data, separators=(",", ":"))
        return orders, conversions, next_trader_data

    def resolve_symbol(self, state: TradingState, names: list[str]) -> Symbol | None:
        name_set = set(names)

        for symbol in state.order_depths.keys():
            if symbol in name_set:
                return symbol

        for symbol, listing in state.listings.items():
            product = listing["product"] if isinstance(listing, dict) else listing.product
            if product in name_set and symbol in state.order_depths:
                return symbol

        return None

    def trade_emerald(self, state: TradingState, symbol: Symbol) -> list[Order]:
        fair_value = 10000.0
        return self.market_make(
            state=state,
            symbol=symbol,
            fair_value=fair_value,
            limit=self.limits["EMERALD"],
            aggressive_edge=1.0,
            passive_edge=1.0,
            inventory_skew=0.20,
        )

    def trade_tomato(
        self,
        state: TradingState,
        symbol: Symbol,
        ema: float | None,
        prev_mid: float | None,
    ) -> tuple[list[Order], float, float]:
        order_depth = state.order_depths[symbol]
        best_bid, best_ask, _, _ = self.best_prices(order_depth)

        if best_bid is None or best_ask is None:
            fallback = ema if ema is not None else 5000.0
            return [], fallback, prev_mid if prev_mid is not None else fallback

        mid = (best_bid + best_ask) / 2.0
        alpha = 0.2
        ema = mid if ema is None else (alpha * mid + (1.0 - alpha) * ema)

        # Tutorial tomato data shows short-horizon anti-momentum. Lean against the last move.
        if prev_mid is None:
            fair_value = ema
        else:
            fair_value = ema - 0.7 * (mid - prev_mid)

        orders = self.market_make(
            state=state,
            symbol=symbol,
            fair_value=fair_value,
            limit=self.limits["TOMATO"],
            aggressive_edge=1.0,
            passive_edge=1.0,
            inventory_skew=0.18,
        )

        return orders, ema, mid

    def market_make(
        self,
        state: TradingState,
        symbol: Symbol,
        fair_value: float,
        limit: int,
        aggressive_edge: float,
        passive_edge: float,
        inventory_skew: float,
    ) -> list[Order]:
        order_depth = state.order_depths[symbol]
        best_bid, best_ask, best_bid_volume, best_ask_volume = self.best_prices(order_depth)

        if best_bid is None or best_ask is None:
            return []

        orders: list[Order] = []
        position = state.position.get(symbol, 0)

        adjusted_fair = fair_value - inventory_skew * position

        buy_capacity = limit - position
        sell_capacity = limit + position

        # Aggressively take mispriced levels, not just top of book.
        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if buy_capacity <= 0:
                break
            if ask_price <= adjusted_fair - aggressive_edge:
                take_qty = min(buy_capacity, -ask_volume)
                if take_qty > 0:
                    orders.append(Order(symbol, ask_price, take_qty))
                    buy_capacity -= take_qty

        for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if sell_capacity <= 0:
                break
            if bid_price >= adjusted_fair + aggressive_edge:
                take_qty = min(sell_capacity, bid_volume)
                if take_qty > 0:
                    orders.append(Order(symbol, bid_price, -take_qty))
                    sell_capacity -= take_qty

        # If we are glued to limits, force partial liquidation so we can quote both sides again.
        if position >= int(0.8 * limit) and sell_capacity > 0:
            extra = max(1, sell_capacity // 2)
            orders.append(Order(symbol, max(best_bid, int(round(adjusted_fair))), -extra))
            sell_capacity -= extra
        elif position <= -int(0.8 * limit) and buy_capacity > 0:
            extra = max(1, buy_capacity // 2)
            orders.append(Order(symbol, min(best_ask, int(round(adjusted_fair))), extra))
            buy_capacity -= extra

        bid_quote = min(int(round(adjusted_fair - passive_edge)), best_bid + 1)
        ask_quote = max(int(round(adjusted_fair + passive_edge)), best_ask - 1)

        if bid_quote >= ask_quote:
            bid_quote = int(round(adjusted_fair - 1))
            ask_quote = int(round(adjusted_fair + 1))

        if buy_capacity > 0:
            orders.append(Order(symbol, bid_quote, buy_capacity))

        if sell_capacity > 0:
            orders.append(Order(symbol, ask_quote, -sell_capacity))

        return orders

    def best_prices(self, order_depth: OrderDepth):
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return None, None, None, None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_volume = order_depth.buy_orders[best_bid]
        best_ask_volume = order_depth.sell_orders[best_ask]

        return best_bid, best_ask, best_bid_volume, best_ask_volume
