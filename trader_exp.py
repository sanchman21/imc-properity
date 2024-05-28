import json
from datamodel import Listing, Observation, Order, Position, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import jsonpickle
import collections
import copy
from collections import defaultdict
import math

class Logger:
	def __init__(self) -> None:
		self.logs = ""
		self.max_log_length = 3750

	def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
		self.logs += sep.join(map(str, objects)) + end

	def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
		base_length = len(self.to_json([
			self.compress_state(state, ""),
			self.compress_orders(orders),
			conversions,
			"",
			"",
		]))

		# We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
		max_item_length = (self.max_log_length - base_length) // 3

		print(self.to_json([
			self.compress_state(state, self.truncate(state.traderData, max_item_length)),
			self.compress_orders(orders),
			conversions,
			self.truncate(trader_data, max_item_length),
			self.truncate(self.logs, max_item_length),
		]))

		self.logs = ""

	def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
		return [
			state.timestamp,
			trader_data,
			self.compress_listings(state.listings),
			self.compress_order_depths(state.order_depths),
			self.compress_trades(state.own_trades),
			self.compress_trades(state.market_trades),
			state.position,
			self.compress_observations(state.observations),
		]

	def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
		compressed = []
		for listing in listings.values():
			compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

		return compressed

	def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
		compressed = {}
		for symbol, order_depth in order_depths.items():
			compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

		return compressed

	def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
		compressed = []
		for arr in trades.values():
			for trade in arr:
				compressed.append([
						trade.symbol,
						trade.price,
						trade.quantity,
						trade.buyer,
						trade.seller,
						trade.timestamp,
				])

		return compressed

	def compress_observations(self, observations: Observation) -> list[Any]:
		conversion_observations = {}
		for product, observation in observations.conversionObservations.items():
			conversion_observations[product] = [
					observation.bidPrice,
					observation.askPrice,
					observation.transportFees,
					observation.exportTariff,
					observation.importTariff,
					observation.sunlight,
					observation.humidity,
			]

		return [observations.plainValueObservations, conversion_observations]

	def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
		compressed = []
		for arr in orders.values():
			for order in arr:
				compressed.append([order.symbol, order.price, order.quantity])

		return compressed

	def to_json(self, value: Any) -> str:
		return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

	def truncate(self, value: str, max_length: int) -> str:
		if len(value) <= max_length:
			return value

		return value[:max_length - 3] + "..."

logger = Logger()

POSITION_LIMITS = {
	"AMETHYSTS": 20,
	"STARFRUIT": 20,
	"ORCHIDS": 100,
	"CHOCOLATE": 232,
	"STRAWBERRIES": 348,
	"ROSES": 58,
	"GIFT_BASKET": 58,
	"COCONUT": 300,
	"COCONUT_COUPON": 600,
}

POSITION_MOMENTUM = {
	"CHOCOLATE": 250,
	"STRAWBERRIES": 350,
	"ROSES": 60
}

DEFAULT_PRICES = {
	"AMETHYSTS": 10_000,
	"STARFRUIT": 5_000,
	"ORCHIDS": 1_000,
	"CHOCOLATE": 8_000,
	"STRAWBERRIES": 4_000,
	"ROSES": 15_000,
	"GIFT_BASKET": 71_355,
	"COCONUT": 1_000,
	"COCONUT_COUPON": 637.5,
}
DATA_MAX_SIZE = 1010
empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'COCONUT' : 0, 'COCONUT_COUPON' : 0, 'ORCHIDS' : 0, 'ROSES' : 0, 'CHOCOLATE' : 0, 'STRAWBERRIES' : 0, 'GIFT_BASKET' : 0}

def def_value():
	return copy.deepcopy(empty_dict)

class Trader:

	person_position = defaultdict(def_value)
	person_actvalof_position = defaultdict(def_value)

	buy_roses = False
	sell_roses = False
	close_roses = False

	buy_strawberries = False
	sell_strawberries = False
	close_strawberries = False

	buy_chocolates = False
	sell_chocolates = False
	close_chocolates = False

	position = copy.deepcopy(empty_dict)
	volume_traded = copy.deepcopy(empty_dict)

	person_position = defaultdict(def_value)
	person_actvalof_position = defaultdict(def_value)

	coconuts_cache = []
	coconuts_dim = 3

	cpnl = defaultdict(lambda : 0)
	steps_coconut = 0
	steps_roses = 0

	def __init__(self):
				self.trader_data = {
			"isResetted": True,
			"historicalPrice": {
				"AMETHYSTS": [],
				"STARFRUIT": [],
				"CHOCOLATE": [],
				"STRAWBERRIES": [],
				"ROSES": [],
				"GIFT_BASKET": [],
				"COCONUT": [],
				"COCONUT_COUPON": [],
				"ORCHIDS": [],
			},
			"custom_basket": {
				"hedge_ratio": 1,
				"position": {
					"gift_basket": 0,
					"chocolates": 0,
					"strawberries": 0,
					"roses": 0,
				},
				"stop_loss": 0,
				"mean": 394,
				"std": 68,
			},

			"coconuts": {
				"mean": 17.236074037725523,
				"std": 0.27935197810524515,
				"sellorbuyMore": False,
				"position": {
					"coconuts": 0,
					"coconut_coupon": 0,
				},
			},
			"orchids" : {
				"hum_min": 59.99958,
				"hum_max": 97.51327, 
				"orchids_min": 960.75, 
				"orchids_max": 1257.25,
				"positions": {
					"short": 0,
					"short_arbit": 0,
				},
			},
			"positionHeldFor": 0,
			"storage_cost": 0.1,
			"historicalBidSouth": [],
			"historicalAskSouth": [],
			"sellorbuyMore": False,
		}

	def get_position(self, product: str, state: TradingState) -> Position: # Get Current Position
		try:
			return state.position[product]
		except KeyError:
			return Position(0)

	def decodeData(self, state: TradingState): # Decode all TraderData
		if state.traderData == "":
			self.trader_data["isResetted"] = False
		if self.trader_data["isResetted"] and state.traderData != "":
			logger.print("Resetting...")
			self.trader_data = jsonpickle.decode(state.traderData)
			self.trader_data["isResetted"] = False     

	def encodeData(self, state: TradingState): # Encode all TraderData
		for product in state.listings.keys():
			for array_var_name in ["historicalPrice"]:
				self.trader_data[array_var_name][product] = self.trader_data[array_var_name][product][-DATA_MAX_SIZE:]
		for array_var_name in ["historicalBidSouth", "historicalAskSouth"]:
			self.trader_data[array_var_name] = self.trader_data[array_var_name][-DATA_MAX_SIZE:]
		trader_data = jsonpickle.encode(self.trader_data)
		return trader_data

	def updateNewData(self, state: TradingState): # Update the TraderData with new data
		for product in state.listings.keys():
			# Updating the mid_price
			mid_price: float = self.get_mid_price(product, state)
			self.trader_data["historicalPrice"][product].append(mid_price)
			if len(self.trader_data["historicalPrice"][product]) > DATA_MAX_SIZE:
				self.trader_data["historicalPrice"][product] = self.trader_data["historicalPrice"][product][-DATA_MAX_SIZE:]

	def get_mid_price(self, product: str, state : TradingState, bidId: int = 0, askId: int = 0): # Get Mid Price (Best Ask + Best Bid) / 2
		default_price = DEFAULT_PRICES[product]
		if product not in state.order_depths:
			return default_price
		market_bids = state.order_depths[product].buy_orders
		if len(market_bids) == 0:
			return self.trader_data["historicalPrice"][product][-1]
		market_asks = state.order_depths[product].sell_orders
		if len(market_asks) == 0:
			return self.trader_data["historicalPrice"][product][-1]
		best_bid = sorted(market_bids, reverse=True)[bidId]
		best_ask = sorted(market_asks)[askId]
		return (best_bid + best_ask)/2

	def get_vwap(self, product: str, state: TradingState):
		order_depths: OrderDepth = state.order_depths[product]
		prices = np.array(
			list(order_depths.buy_orders.keys()) +
			list(order_depths.sell_orders.keys()))
		volumes = np.array(
			list(order_depths.buy_orders.values()) +
			list(map(abs, list(order_depths.sell_orders.values()))))
		vwap = np.sum(prices * volumes) / np.sum(volumes)
		return vwap
	
	def do_regression(self, product: str, window: int = 5, time_period: int = 100):
		prices = self.trader_data["historicalPrice"][product][-time_period-2*window:-window+1]
		def create_features_and_target(data, window_size):
				X, y = [], []
				for i in range(len(data) - window_size):
						X.append(data[i:i+window_size])
						y.append(data[i+window_size])
				return np.array(X), np.array(y)
		X, y = create_features_and_target(prices, window)
		one = np.ones((len(X),1))
		X = np.append(one, X, axis=1)
		#reshape Y to a column vector
		y = np.array(y).reshape((len(y),1))
		beta = np.dot((np.linalg.inv(np.dot(X.T,X))), np.dot(X.T,y))
		return np.squeeze(beta).tolist()

	def lr_starfruit(self, product: str, state: TradingState, window: int = 5, time_period: int = 100): # Linear Regression
		coefs = [0.1846303394002591, 0.21191493805440884, 0.2629990148836522, 0.3400095834526976]
		intercept = 2.2484060649790805
		historical_price_data: List[int] = self.trader_data["historicalPrice"][product]
		n = len(historical_price_data)
		if n < window:
				return self.get_vwap(product, state)
		if n % (time_period)/2 == 0 and n > time_period:
				beta = self.do_regression(product, window=window, time_period=time_period)
				coefs = beta[1:]
				intercept = beta[0]
		prices_cache = historical_price_data[-window:]
		nxt_price = intercept
		for i, coef in enumerate(coefs):
			nxt_price += coef*prices_cache[i]
		return int(round(nxt_price))
	
	def get_max_orders(self, product: str, state: TradingState, side: str, max_volume: int):
		order_depth: OrderDepth = state.order_depths[product]
		orders = []
		if side == "buy":
			for price, volume in order_depth.sell_orders.items():
				if max_volume <= 0:
					break
				order_volume = min(max_volume, -volume)
				orders.append(Order(product, price, int(order_volume)))
				max_volume -= order_volume
		elif side == "sell":
			for price, volume in order_depth.buy_orders.items():
				if max_volume >= 0:
					break
				order_volume = -min(-max_volume, volume)
				orders.append(Order(product, price, int(order_volume)))
				max_volume -= order_volume
		return orders
	
	def get_ema(self, product: str, window: int):
		prices = pd.Series(self.trader_data["historicalPrice"][product][-window:])
		return prices.ewm(span=window, adjust=False).mean().iloc[-1]

	def compute_max_orchids_short(self, state: TradingState):
		order_depth: OrderDepth = state.order_depths["ORCHIDS"]
		order_volumes = list(order_depth.buy_orders.values())
		volume = 0
		max_short_volume = POSITION_LIMITS["ORCHIDS"]+self.get_position("ORCHIDS", state)
		for level in range(len(order_volumes)):
			volume += order_volumes[level]
		volume = -min(volume, max_short_volume)
		return volume
	
	def get_past_x_ema(self, product, window):
		prices = pd.Series(self.trader_data["historicalPrice"][product][-window:])
		emas = list(prices.ewm(span=window, adjust=False).mean())
		return emas[0]

	def strat_amethysts(self, state: TradingState):
		product: str = "AMETHYSTS"
		order_depth: OrderDepth = state.order_depths[product]
		order: List[Order] = []
		curr_pos: int = self.get_position(product, state)
		new_curr_pos: int = curr_pos
		max_ask_volume = -curr_pos - POSITION_LIMITS[product]
		max_bid_volume = -curr_pos + POSITION_LIMITS[product] 
		predicted_next_price: int = DEFAULT_PRICES[product]

		sellId: int = 0
		buyId: int = 0

		# MARKET TAKING
		if len(order_depth.sell_orders) != 0:
			best_ask, best_ask_amount = list(order_depth.sell_orders.items())[sellId]
			if best_ask < 9999: # Signals / Buying Conditions
				logger.print(f"BUY_{product}", str(-best_ask_amount) + "x", best_ask)
				order.append(Order(product, best_ask, min(-best_ask_amount, max_bid_volume))) 
				logger.print("Best Ask Volume:", best_ask_amount)
				logger.print("Max Bid Volume:", max_bid_volume)
				logger.print("Buy:", min(-best_ask_amount, max_bid_volume))
				new_curr_pos = curr_pos + min(-best_ask_amount, max_bid_volume)
				logger.print("New Position:", new_curr_pos)
				sellId += 1

		if len(order_depth.buy_orders) != 0:
			best_bid, best_bid_amount = list(order_depth.buy_orders.items())[buyId]
			if best_bid > 10001: # Signals / Selling Conditions
				logger.print(f"SELL_{product}", str(best_bid_amount) + "x", best_bid)
				order.append(Order(product, best_bid, max(-best_bid_amount, max_ask_volume))) 
				logger.print("Best Bid Volume:", best_bid_amount)
				logger.print("Max Ask Volume:", max_ask_volume)
				logger.print("Sell:", max(-best_bid_amount, max_ask_volume))
				new_curr_pos = new_curr_pos + max(-best_bid_amount, max_ask_volume)
				logger.print("New Position:", new_curr_pos)
				buyId += 1

		# MARKET MAKING
		try:
			best_bid = sorted(set(order_depth.buy_orders.keys()), reverse=True)[buyId]
			buy_pr = min(best_bid + 1, predicted_next_price-1)
		except:
			buy_pr = predicted_next_price-1
		try:
			best_ask = sorted(set(order_depth.sell_orders.keys()))[sellId]
			sell_pr = max(best_ask - 1, predicted_next_price+1)
		except:
			sell_pr = predicted_next_price+1
		best_bid = list(order_depth.buy_orders.keys())[0]
		best_ask = list(order_depth.sell_orders.keys())[0]
		if best_ask - best_bid > 4:
			buy_pr = best_bid + 1
			sell_pr = best_ask - 1
		bid_volume = min(POSITION_LIMITS[product]-new_curr_pos,POSITION_LIMITS[product]-curr_pos)
		order.append(Order(product,buy_pr,bid_volume))
		ask_volume = max(-POSITION_LIMITS[product]-new_curr_pos,-POSITION_LIMITS[product]-curr_pos)
		order.append(Order(product,sell_pr,ask_volume))

		return order

	def strat_starfruit(self, state: TradingState):
		product: str = "STARFRUIT"
		order_depth: OrderDepth = state.order_depths[product]
		order: List[Order] = []
		curr_pos: int = self.get_position(product, state)
		new_curr_pos: int = curr_pos
		max_ask_volume = -curr_pos - POSITION_LIMITS[product]
		max_bid_volume = -curr_pos + POSITION_LIMITS[product] 
		regression_price: int = self.lr_starfruit(product, state, window=4, time_period=100)
		vwap = self.get_vwap(product, state)

		sellId: int = 0
		buyId: int = 0

		# MARKET TAKING
		if len(order_depth.sell_orders) != 0:
			best_ask, best_ask_amount = list(order_depth.sell_orders.items())[sellId]
			if best_ask < vwap: # Signals / Buying Conditions
					logger.print(f"BUY_{product}", str(-best_ask_amount) + "x", best_ask)
					order.append(Order(product, best_ask, min(-best_ask_amount, max_bid_volume))) 
					logger.print("Best Ask Volume:", best_ask_amount)
					logger.print("Max Bid Volume:", max_bid_volume)
					logger.print("Buy:", min(-best_ask_amount, max_bid_volume))
					new_curr_pos = curr_pos + min(-best_ask_amount, max_bid_volume)
					logger.print("New Position:", new_curr_pos)
					sellId += 1

		if len(order_depth.buy_orders) != 0:
			best_bid, best_bid_amount = list(order_depth.buy_orders.items())[buyId]
			if best_bid > vwap: # Signals / Selling Conditions
				logger.print(f"SELL_{product}", str(best_bid_amount) + "x", best_bid)
				order.append(Order(product, best_bid, max(-best_bid_amount, max_ask_volume))) 
				logger.print("Best Bid Volume:", best_bid_amount)
				logger.print("Max Ask Volume:", max_ask_volume)
				logger.print("Sell:", max(-best_bid_amount, max_ask_volume))
				new_curr_pos = new_curr_pos + max(-best_bid_amount, max_ask_volume)
				logger.print("New Position:", new_curr_pos)
				buyId += 1

				# MARKET MAKING
		try:
			best_bid = sorted(set(order_depth.buy_orders.keys()), reverse=True)[buyId]
			buy_pr = min(best_bid + 2, int(regression_price-2))
		except:
			buy_pr = int(regression_price-2)
		try:
			best_ask = sorted(set(order_depth.sell_orders.keys()))[sellId]
			sell_pr = max(best_ask - 2, int(regression_price+2))
		except:
			sell_pr = int(regression_price+2)
		best_bid = list(order_depth.buy_orders.keys())[0]
		best_ask = list(order_depth.sell_orders.keys())[0]
		if best_ask - best_bid > 4:
			buy_pr = best_bid + 1
			sell_pr = best_ask - 1
		bid_volume = min(POSITION_LIMITS[product]-new_curr_pos,POSITION_LIMITS[product]-curr_pos)
		order.append(Order(product,buy_pr,bid_volume))
		ask_volume = max(-POSITION_LIMITS[product]-new_curr_pos,-POSITION_LIMITS[product]-curr_pos)
		order.append(Order(product,sell_pr,ask_volume))
		return order
	
	def get_exit_all_orders(self, state: TradingState, position: str):
		chocs, straws, roses, gb = 0, 0, 0, 0
		if position == "long":
			chocs_orders = list(state.order_depths["CHOCOLATE"].sell_orders.values())
			straws_orders = list(state.order_depths["STRAWBERRIES"].sell_orders.values())
			roses_orders = list(state.order_depths["ROSES"].sell_orders.values())
			gb_orders = list(state.order_depths["GIFT_BASKET"].buy_orders.values())
			for level in range(len(chocs_orders)):
				chocs -= chocs_orders[level]
			for level in range(len(straws_orders)):
				straws -= straws_orders[level]
			for level in range(len(roses_orders)):
				roses -= roses_orders[level]
			for level in range(len(gb_orders)):
				gb += gb_orders[level]
			chocs = min(self.trader_data["custom_basket"]["position"]["chocolates"], chocs)
			straws = min(self.trader_data["custom_basket"]["position"]["strawberries"], straws)
			roses = min(self.trader_data["custom_basket"]["position"]["roses"], roses)
			gb = min(-self.trader_data["custom_basket"]["position"]["gift_basket"], gb)
			self.trader_data["custom_basket"]["position"]["chocolates"] -= chocs
			self.trader_data["custom_basket"]["position"]["strawberries"] -= straws
			self.trader_data["custom_basket"]["position"]["roses"] -= roses
			self.trader_data["custom_basket"]["position"]["gift_basket"] += gb
			return chocs, straws, roses, gb
		else:
			chocs_orders = list(state.order_depths["CHOCOLATE"].buy_orders.values())
			straws_orders = list(state.order_depths["STRAWBERRIES"].buy_orders.values())
			roses_orders = list(state.order_depths["ROSES"].buy_orders.values())
			gb_orders = list(state.order_depths["GIFT_BASKET"].sell_orders.values())
			for level in range(len(chocs_orders)):
				chocs += chocs_orders[level]
			for level in range(len(straws_orders)):
				straws += straws_orders[level]
			for level in range(len(roses_orders)):
				roses += roses_orders[level]
			for level in range(len(gb_orders)):
				gb -= gb_orders[level]
			chocs = min(-self.trader_data["custom_basket"]["position"]["chocolates"], chocs)
			straws = min(-self.trader_data["custom_basket"]["position"]["strawberries"], straws)
			roses = min(-self.trader_data["custom_basket"]["position"]["roses"], roses)
			gb = min(self.trader_data["custom_basket"]["position"]["gift_basket"], gb)
			self.trader_data["custom_basket"]["position"]["chocolates"] += chocs
			self.trader_data["custom_basket"]["position"]["strawberries"] += straws
			self.trader_data["custom_basket"]["position"]["roses"] += roses
			self.trader_data["custom_basket"]["position"]["gift_basket"] -= gb
			return chocs, straws, roses, gb
	
	# compute the maximum number of orders that can be placed for each product in the basket
	def compute_max_orders_long(self, state: TradingState, side: str):
		chocs, straws, roses, gb = 0, 0, 0, 0 # set initial orders to 0
		print(state.order_depths["CHOCOLATE"].sell_orders.items())
		print(state.order_depths["STRAWBERRIES"].sell_orders.items())
		print(state.order_depths["ROSES"].sell_orders.items())
		print(state.order_depths["GIFT_BASKET"].buy_orders.items())
		hedge_ratio = self.trader_data["custom_basket"]["hedge_ratio"] # set hedge ratio
		# if long (buy) CUSTOM BASKET and short (sell) GIFT BASKET (initiate position)
		chocs_orders = list(state.order_depths["CHOCOLATE"].sell_orders.values())
		straws_orders = list(state.order_depths["STRAWBERRIES"].sell_orders.values())
		roses_orders = list(state.order_depths["ROSES"].sell_orders.values())
		gb_orders = list(state.order_depths["GIFT_BASKET"].buy_orders.values())
		for level in range(len(chocs_orders)): # for each level of buy orders in the custom basket
			chocs -= chocs_orders[level] # subtract the order volume from the current position
		for level in range(len(straws_orders)): # for each level of buy orders in the custom basket
			straws -= straws_orders[level]
		for level in range(len(roses_orders)):
			roses -= roses_orders[level]
		for level in range(len(gb_orders)): # for each level of sell orders in the gift basket # if product is gift basket
			gb += gb_orders[level] # add the order volume to the current position (this variable will be positive)
		chocs = min(POSITION_LIMITS["CHOCOLATE"], chocs) # set the maximum order volume for chocolates
		straws = min(POSITION_LIMITS["STRAWBERRIES"], straws) # set the maximum order volume for strawberries
		roses = min(POSITION_LIMITS["GIFT_BASKET"], roses) # set the maximum order volume for roses
		gb = min(POSITION_LIMITS["GIFT_BASKET"], gb) # set the maximum order volume for gift basket (short)

		chocs = chocs - (chocs % 4)
		straws = straws - (straws % 6)
		if gb == 0:
			return 0, 0, 0, 0
		else:
			roses = min(chocs/4, straws/6, roses, gb)
			chocs, straws, gb = 4*roses, 6*roses, roses
		self.trader_data["custom_basket"]["position"]["chocolates"] += chocs
		self.trader_data["custom_basket"]["position"]["strawberries"] += straws
		self.trader_data["custom_basket"]["position"]["roses"] += roses
		self.trader_data["custom_basket"]["position"]["gift_basket"] -= gb
		return chocs, straws, roses, gb # returns positive volumes even for sell orders
	
	# compute the maximum number of orders that can be placed for each product in the basket
	def compute_max_orders_short(self, state: TradingState, side: str):
		chocs, straws, roses, gb = 0, 0, 0, 0 # set initial orders to 0
		hedge_ratio = self.trader_data["custom_basket"]["hedge_ratio"] # set hedge ratio
		# if short (sell) CUSTOM BASKET and long (buy) GIFT BASKET (initiate position)
		chocs_orders = list(state.order_depths["CHOCOLATE"].buy_orders.values())
		straws_orders = list(state.order_depths["STRAWBERRIES"].buy_orders.values())
		roses_orders = list(state.order_depths["ROSES"].buy_orders.values())
		gb_orders = list(state.order_depths["GIFT_BASKET"].sell_orders.values())
		for level in range(len(chocs_orders)): # for each level of buy orders in the custom basket
			chocs += chocs_orders[level] # subtract the order volume from the current position
		for level in range(len(straws_orders)): # for each level of buy orders in the custom basket
			straws += straws_orders[level]
		for level in range(len(roses_orders)):
			roses += roses_orders[level]
		for level in range(len(gb_orders)): # for each level of sell orders in the gift basket # if product is gift basket
			gb -= gb_orders[level] # add the order volume to the current position (this variable will be positive)
		chocs = min(POSITION_LIMITS["CHOCOLATE"], chocs)
		straws = min(POSITION_LIMITS["STRAWBERRIES"], straws)
		roses = min(POSITION_LIMITS["ROSES"], roses)
		gb = min(POSITION_LIMITS["GIFT_BASKET"], gb)
			
		chocs = chocs - (chocs % 4)
		straws = straws - (straws % 6)
		if gb == 0:
			return 0, 0, 0, 0
		else:
			roses = min(chocs/4, straws/6, roses, gb)
			chocs, straws, gb = 4*roses, 6*roses, roses
		self.trader_data["custom_basket"]["position"]["chocolates"] -= chocs
		self.trader_data["custom_basket"]["position"]["strawberries"] -= straws
		self.trader_data["custom_basket"]["position"]["roses"] -= roses
		self.trader_data["custom_basket"]["position"]["gift_basket"] += gb
		return chocs, straws, roses, gb # returns positive volumes even for sell orders

	# get individual orders (per product) for the basket strategy
	def get_orders(self, state: TradingState, product: str, side: str, max_volume: int):
		order_depth: OrderDepth = state.order_depths[product]
		orders = []
		if side == "buy":
			for price, volume in order_depth.sell_orders.items():
				if max_volume <= 0:
					break
				order_volume = min(max_volume, -volume)
				print("PRODUCT: ", product, "PRICE: ", price, "VOLUME: ", order_volume)
				orders.append(Order(product, price, int(order_volume)))
				max_volume -= order_volume
		elif side == "sell":
			for price, volume in order_depth.buy_orders.items():
				if max_volume >= 0:
					break
				order_volume = -min(-max_volume, volume)
				print("PRODUCT: ", product, "PRICE: ", price, "VOLUME: ", order_volume)
				orders.append(Order(product, price, int(order_volume)))
				max_volume -= order_volume
		return orders
	
	def get_arbit_spread(self):
		choc = self.trader_data["historicalPrice"]["CHOCOLATE"][-1]*4
		straw = self.trader_data["historicalPrice"]["STRAWBERRIES"][-1]*6
		rose = self.trader_data["historicalPrice"]["ROSES"][-1]
		gb = self.trader_data["historicalPrice"]["GIFT_BASKET"][-1]
		custom_basket = choc + straw + rose
		return gb - custom_basket
	
	def compute_new_mean_dev(self):
		mean, std = self.trader_data["custom_basket"]["mean"], self.trader_data["custom_basket"]["std"]
		n = 1000
		spread = self.get_arbit_spread()
		new_mean = mean + (spread - mean) / n
		new_std = np.sqrt((std**2 * (n - 1) + (spread - mean)**2) / n)
		self.trader_data["custom_basket"]["mean"], self.trader_data["custom_basket"]["std"] = new_mean, new_std
		return new_mean, new_std

	def strat_basket(self, state: TradingState):
		mean, std = self.compute_new_mean_dev()
		spread = self.get_arbit_spread()
		orders = {"GIFT_BASKET": [], "CHOCOLATE": [], "STRAWBERRIES": [], "ROSES": []}
		real_pos_gb = self.get_position("GIFT_BASKET", state)
		max_real_gb_ask_volume = -real_pos_gb - POSITION_LIMITS["GIFT_BASKET"]
		max_real_gb_bid_volume = -real_pos_gb + POSITION_LIMITS["GIFT_BASKET"]

		if spread > 469:
			print("SPREAD: ", spread, "ENTRY SIGNAL", mean - std)
			# long custom basket, short gift basket
			chocs, straws, roses, gb = self.compute_max_orders_long(state, "buy")
			orders["GIFT_BASKET"] = self.get_orders(state, "GIFT_BASKET", "sell", max_real_gb_ask_volume)
			return orders
		
		if spread < 241:
			# short custom basket, long gift basket
			chocs, straws, roses, gb = self.compute_max_orders_short(state, "sell")
			orders["GIFT_BASKET"] = self.get_orders(state, "GIFT_BASKET", "buy", max_real_gb_bid_volume)
			return orders
	
	def compute_new_mean_dev_cocs(self):
		n = 1000
		mean, std = self.trader_data["coconuts"]["mean"], self.trader_data["coconuts"]["std"]
		if mean == 0 and len(self.trader_data["historicalPrice"]["COCONUT"]) >= n:
			prev_coconut_prices = self.trader_data["historicalPrice"]["COCONUT"][-n:]
			prev_coconut_coupon_prices = self.trader_data["historicalPrice"]["COCONUT_COUPON"][-n:]
			spread = np.array(prev_coconut_prices) / np.array(prev_coconut_coupon_prices)
			mean = spread[-1]
			std = spread.std()
			self.trader_data["coconuts"]["mean"], self.trader_data["coconuts"]["std"] = mean, std
			return mean, std
		spread = self.trader_data["historicalPrice"]["COCONUT"][-1] / self.trader_data["historicalPrice"]["COCONUT_COUPON"][-1]
		new_mean = mean + (spread - mean) / n
		new_std = np.sqrt((std**2 * (n - 1) + (spread - mean)**2) / n)
		self.trader_data["coconuts"]["mean"], self.trader_data["coconuts"]["std"] = new_mean, new_std
		return new_mean, new_std

	def strat_coconuts(self, state: TradingState):
			orders = {"COCONUT": [], "COCONUT_COUPON": []}
			pos_coupons = self.get_position("COCONUT_COUPON", state)
			max_coupons_bid_volume = -pos_coupons + POSITION_LIMITS["COCONUT_COUPON"]
			max_coupons_ask_volume = -pos_coupons - POSITION_LIMITS["COCONUT_COUPON"]
			mean, std = self.compute_new_mean_dev_cocs()
			spread = self.trader_data["historicalPrice"]["COCONUT"][-1] / self.trader_data["historicalPrice"]["COCONUT_COUPON"][-1]

			if spread < mean - 0.5*std:
				# long coconuts, short coupons
				print("Going Long")
				orders["COCONUT_COUPON"] = self.get_orders(state, "COCONUT_COUPON", "sell", max_coupons_ask_volume)
				return orders
			
			if spread > mean + 0.5*std:
				# short coconuts, long coconut coupons
				print("Going Short")
				orders["COCONUT_COUPON"] = self.get_orders(state, "COCONUT_COUPON", "buy", max_coupons_bid_volume)
				return orders
	
	def compute_orders_basket_items(self, order_depth: Dict[str, OrderDepth]):
		orders = { 'GIFT_BASKET': [], 'STRAWBERRIES': [], 'CHOCOLATE' : [], "ROSES": []}
		prods = {'GIFT_BASKET', 'STRAWBERRIES', 'CHOCOLATE', "ROSES"}
		best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}
		
		osell: Dict[str, Dict[int, int]] = {}
		obuy: Dict[str, Dict[int, int]] = {}

		for p in prods:
			osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
			obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

			best_sell[p] = next(iter(osell[p]))
			best_buy[p] = next(iter(obuy[p]))

			worst_sell[p] = next(reversed(osell[p]))
			worst_buy[p] = next(reversed(obuy[p]))

			mid_price[p] = (best_sell[p] + best_buy[p])/2
			vol_buy[p], vol_sell[p] = 0, 0
			for price, vol in obuy[p].items():
				vol_buy[p] += vol 
			for price, vol in osell[p].items():
				vol_sell[p] += -vol

		if int(round(self.person_position['Rhianna']['ROSES'])) < 0:
			self.buy_roses = False
			self.sell_roses = True
		if int(round(self.person_position['Rhianna']['ROSES'])) > 0:
			self.sell_roses = False
			self.buy_roses = True

		if self.buy_roses and self.position['ROSES'] == POSITION_LIMITS['ROSES']:
			self.buy_roses = False
		if self.sell_roses and self.position['ROSES'] == -POSITION_LIMITS['ROSES']:
			self.sell_roses = False
		if self.close_roses and self.position['ROSES'] == 0:
			self.close_roses = False

		if self.buy_roses:
			vol = POSITION_LIMITS['ROSES'] - self.position['ROSES']
			orders['ROSES'].append(Order('ROSES', best_sell['ROSES'], vol))
		if self.sell_roses:
			vol = self.position['ROSES'] + POSITION_LIMITS['ROSES']
			orders['ROSES'].append(Order('ROSES', best_buy['ROSES'], -vol))
		if self.close_roses:
			vol = -self.position['ROSES']
			if vol < 0:
				orders['ROSES'].append(Order('ROSES', best_buy['ROSES'], vol)) 
			else:
				orders['ROSES'].append(Order('ROSES', best_sell['ROSES'], vol)) 

		if int(round(self.person_position['Vladimir']['CHOCOLATE'])) < 0:
			self.buy_chocolates = False
			self.sell_chocolates = True
		if int(round(self.person_position['Vladimir']['CHOCOLATE'])) > 0:
			self.sell_chocolates = False
			self.buy_chocolates = True

		if self.buy_chocolates and self.position['CHOCOLATE'] == POSITION_LIMITS['CHOCOLATE']:
			self.buy_chocolates = False
		if self.sell_chocolates and self.position['CHOCOLATE'] == -POSITION_LIMITS['CHOCOLATE']:
			self.sell_chocolates = False
		if self.close_chocolates and self.position['CHOCOLATE'] == 0:
			self.close_chocolates = False

		if self.buy_chocolates:
			vol = POSITION_LIMITS['CHOCOLATE'] - self.position['CHOCOLATE']
			orders['CHOCOLATE'].append(Order('CHOCOLATE', best_sell['CHOCOLATE'], vol))
		if self.sell_chocolates:
			vol = self.position['CHOCOLATE'] + POSITION_LIMITS['CHOCOLATE']
			orders['CHOCOLATE'].append(Order('CHOCOLATE', best_buy['CHOCOLATE'], -vol))
		if self.close_chocolates:
			vol = -self.position['CHOCOLATE']
			if vol < 0:
				orders['CHOCOLATE'].append(Order('CHOCOLATE', best_buy['CHOCOLATE'], vol)) 
			else:
				orders['CHOCOLATE'].append(Order('CHOCOLATE', best_sell['CHOCOLATE'], vol)) 
	
		return orders
	
	def calc_next_price_coconuts(self):
		# roses cache stores price from 1 day ago, current day resp
		# by price, here we mean mid price

		coef = [  0.       ,   -5.53676036  , 4.9752956 ,  89.90516943 ,-12.54153176,
  47.68834418, -19.03548319 ,-53.97953179 , 58.72138235, -20.64415707,
   9.06260685 ,-20.47280247 , -6.65062299 , 21.67690024 , 14.45272465,
   0.23974478 ,  0.54511135 , -9.13652145, -24.68053209 , 14.80181452,
 -27.08700391 , -2.30423749 , 46.80148452 ,  0.15293824 , 21.0370698,
   6.01324878  ,-6.92945018  , 5.8873228 , -31.74549852, -47.21158396,
  -7.37361672 , 26.43741694 , -3.81034131 ,-18.52008758 , 38.51428127,
   3.23435657 , 15.0566931  ,-23.78661316 , 14.99109102 , -3.44414343,
  -8.26465934  ,-0.29253448 , -2.44901089 , -7.62306012 , 17.61265979,
 -15.49022044  ,-3.80579791 ,-10.03408512 , -5.34690184 , 26.41352228,
  -5.93092808  ,19.26554148 , 12.78241407 , -1.11060065 ,-10.41089449,
 -11.41096791]
		intercept = 9998.974010350088
		nxt_price = intercept
		for i, val in enumerate(self.coconuts_cache):
			nxt_price += val * coef[i]

		return int(round(nxt_price))

	def values_extract(self, order_dict, buy=0):
		tot_vol = 0
		best_val = -1
		mxvol = -1

		for ask, vol in order_dict.items():
			if (buy == 0):
				vol *= -1
			tot_vol += vol
			if tot_vol > mxvol:
				mxvol = vol
				best_val = ask

		return tot_vol, best_val

	def compute_orders_coconut(self, product, order_depth, acc_bid, acc_ask, LIMIT):
		orders: list[Order] = []

		osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
		obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

		sell_vol, best_sell_pr = self.values_extract(osell)
		buy_vol, best_buy_pr = self.values_extract(obuy, 1)

		cpos = self.position[product]

		for ask, vol in osell.items():
			if ((ask <= acc_bid) or ((self.position[product] < 0) and (ask == acc_bid + 1))) and cpos < LIMIT:
				order_for = min(-vol, LIMIT - cpos)
				cpos += order_for
				assert (order_for >= 0)
				orders.append(Order(product, ask, order_for))

		undercut_buy = best_buy_pr + 1
		undercut_sell = best_sell_pr - 1

		bid_pr = min(undercut_buy, acc_bid)  # we will shift this by 1 to beat this price
		sell_pr = max(undercut_sell, acc_ask)

		if cpos < LIMIT:
			num = LIMIT - cpos
			orders.append(Order(product, bid_pr, num))
			cpos += num

		cpos = self.position[product]

		for bid, vol in obuy.items():
			if ((bid >= acc_ask) or ((self.position[product] > 0) and (bid + 1 == acc_ask))) and cpos > -LIMIT:
				order_for = max(-vol, -LIMIT - cpos)
				# order_for is a negative number denoting how much we will sell
				cpos += order_for
				assert (order_for <= 0)
				orders.append(Order(product, bid, order_for))

		if cpos > -LIMIT:
			num = -LIMIT - cpos
			orders.append(Order(product, sell_pr, num))
			cpos += num
		return orders
	
	def compute_orders(self, product, order_depth, acc_bid, acc_ask):
		if product == "COCONUT":
			return self.compute_orders_coconut(product, order_depth, acc_bid, acc_ask, POSITION_LIMITS[product])
		
	def calc_metrics_bids(self, price_dict):
		volume = 0 
		highest_bid = 0
		bids_vwap = 0
		print(price_dict)

		#In this function we are looping through the bids from highest (most attractive) to lowest (least attractive)
		#We use this function to find the three most important metrics: total volume of bids, highest (most attractive) bid, and the total vwap of bids
		if len(price_dict) > 0:
			price_dict = collections.OrderedDict(sorted(price_dict.items(), reverse=True))
			for index, (key,value) in enumerate(price_dict.items()):
				if index == 0:
					highest_bid = key
				volume += value
				bids_vwap += key*value
			
			if volume != 0:
				bids_vwap /= volume
			else:
				bids_vwap /= 1

		return volume, highest_bid, bids_vwap
	
	def calc_metrics_asks(self, price_dict): 
		volume = 0 
		lowest_ask = 0
		asks_vwap = 0

		#In this function we are looping through the asks from lowest (most attractive) to highest (least attractive)
		#We use this function to find the three most important metrics: total volume of asks, lowest (most attractive) bid, and the total vwap of asks
		if len(price_dict) > 0:
			price_dict = collections.OrderedDict(sorted(price_dict.items()))
			for index, (key,value) in enumerate(price_dict.items()):
				if index == 0:
					lowest_ask = key
				volume += -value
				asks_vwap += key*-value
			
			if volume != 0:
				asks_vwap /= volume
			else:
				asks_vwap /= 1

		return volume, lowest_ask, asks_vwap

	#ORCHIDS
	def arb_orders_ORCHID(self, state, pos, pos_limit):  

		bid_orders_to_submit: List[Order] = []
		ask_orders_to_submit: List[Order] = []

		order_book              = state.order_depths["ORCHIDS"]
		local_bid_prices       = collections.OrderedDict(sorted(order_book.buy_orders.items(), reverse=True))
		local_ask_prices        = collections.OrderedDict(sorted(order_book.sell_orders.items()))
		buy_volume_avail        = pos_limit - pos
		sell_volume_avail       = abs(-pos_limit - pos)

		south_island_info       = state.observations.conversionObservations["ORCHIDS"]
		adj_south_ask_price     = south_island_info.askPrice + south_island_info.transportFees + south_island_info.importTariff
		adj_south_bid_price    = south_island_info.bidPrice - south_island_info.transportFees - south_island_info.exportTariff

		bLocal_sSouth_pnl = 0
		sLocal_bSouth_pnl = 0
		
		#here we buy local (at the ask), and will sell at the `adj_south_ask_price`
		for local_ask_price, ask_volume in local_ask_prices.items():
			if (local_ask_price < adj_south_bid_price) and buy_volume_avail > 0:
				logger.print("arb: g1 bLocal,sSouth:", local_ask_price, adj_south_ask_price)
				max_vol_tradable = min(abs(ask_volume), buy_volume_avail)
				bid_orders_to_submit.append(Order("ORCHIDS", int(local_ask_price), int(max_vol_tradable)))
				bLocal_sSouth_pnl += (adj_south_bid_price-local_ask_price)*max_vol_tradable
				local_ask_prices[local_ask_price] -= max_vol_tradable
				local_ask_prices = {k: v for k, v in local_ask_prices.items() if k > 0}
				buy_volume_avail -= max_vol_tradable
				
		#here we sell local (at the bid), and will buy at the `adj_south_bid_price`
		for local_bid_price, bid_volume in local_bid_prices.items():
			if (adj_south_ask_price < local_bid_price) and sell_volume_avail > 0:
				logger.print("arb: g2 sLocal,bSouth:", local_bid_price, adj_south_bid_price)
				max_vol_tradable = min(bid_volume, sell_volume_avail)
				ask_orders_to_submit.append(Order("ORCHIDS", int(local_bid_price), int(-max_vol_tradable)))
				sLocal_bSouth_pnl += (local_bid_price-adj_south_ask_price)*max_vol_tradable
				local_bid_prices[local_bid_price] -= max_vol_tradable
				local_bid_prices = {k: v for k, v in local_bid_prices.items() if k > 0}
				sell_volume_avail -= max_vol_tradable

		#Now we decide which side to MM on, bLocal,sSouth or sLocal,bSouth BECAUSE you can only do conversions simalaniously in short/long
				
		#hyperparam
		margin = 1
	
		#if there's sure gains in one direction
		if bLocal_sSouth_pnl > sLocal_bSouth_pnl:
			#buy Local (at ask), sell 
			if buy_volume_avail > 0:
				logger.print(adj_south_bid_price,adj_south_ask_price)
				MM_price = lambda x: x - 1 if math.floor(x) == x else math.floor(x)
				bid_orders_to_submit.append(Order("ORCHIDS", int(MM_price(adj_south_bid_price)), int(buy_volume_avail)))
				return bid_orders_to_submit
		elif bLocal_sSouth_pnl < sLocal_bSouth_pnl:

			if sell_volume_avail > 0:
				logger.print(adj_south_bid_price,adj_south_ask_price)
				MM_price = lambda x: x + 1 if math.ceil(x) == x else math.ceil(x)
				ask_orders_to_submit.append(Order("ORCHIDS", int(MM_price(adj_south_ask_price)), int(-sell_volume_avail)))
				return ask_orders_to_submit
		#if not MM based on closest strat to an arb (both will be negative)
		else:
			_,lowest_local_ask_price,_ = self.calc_metrics_asks(local_ask_prices)
			_,highest_local_bid_price,_ = self.calc_metrics_bids(local_bid_prices)

			logger.print(adj_south_bid_price,adj_south_ask_price)

			bLocal_sSouth_pnl = round(adj_south_bid_price - lowest_local_ask_price, 2)
			sLocal_bSouth_pnl = round(highest_local_bid_price - adj_south_ask_price, 2)

			if bLocal_sSouth_pnl > sLocal_bSouth_pnl: 
				logger.print('sell MM')
				MM_price = lambda x: x - 1 if math.floor(x) == x else math.floor(x)
				return [Order("ORCHIDS", int(MM_price(adj_south_bid_price)), buy_volume_avail)]
			else:
				MM_price = lambda x: x + 1 if math.ceil(x) == x else math.ceil(x)
				logger.print("buy MM")
				return [Order("ORCHIDS", int(MM_price(adj_south_ask_price)), -sell_volume_avail)]
				
	def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
		orders = {"COCONUT": [], "COCONUT_COUPON": [], "ORCHIDS": [], "STARFRUIT": [], "AMETHYSTS": [], "GIFT_BASKET": [], "CHOCOLATE": [], "STRAWBERRIES": [], "ROSES": []}
		conversions = 0
		self.decodeData(state)
		self.updateNewData(state)
		for product in state.listings.keys():
			if product == "AMETHYSTS":
				orders[product] = self.strat_amethysts(state)
			if product == "STARFRUIT":
				orders[product] = self.strat_starfruit(state)
			if product == "GIFT_BASKET":
				basket_orders = self.strat_basket(state)
				if basket_orders:
					orders.update(basket_orders)
			if product == "COCONUT":
				coconut_orders = self.strat_coconuts(state)
				if coconut_orders:
					orders.update(coconut_orders)
			if product == "ORCHIDS":
				current_orch_pos = self.get_position("ORCHIDS") 
				orc_orders = self.arb_orders_ORCHID(state, current_orch_pos, 100)
				conversions = current_orch_pos*-1
				orders[product] = orc_orders

		for key, val in state.position.items():
			self.position[key] = val

		self.steps_roses += 1

		for product in state.market_trades.keys():
			for trade in state.market_trades[product]:
				if trade.buyer == trade.seller:
					continue
				self.person_position[trade.buyer][product] = 1
				self.person_position[trade.seller][product] = -1
				self.person_actvalof_position[trade.buyer][product] += trade.quantity
				self.person_actvalof_position[trade.seller][product] += -trade.quantity

		basket_items_orders = self.compute_orders_basket_items(state.order_depths)
		orders['ROSES'] = basket_items_orders['ROSES']
		orders['CHOCOLATE'] = basket_items_orders['CHOCOLATE']

		if len(self.coconuts_cache) == self.coconuts_dim:
			self.coconuts_cache.pop(0)
		_, bs_coconuts = self.values_extract(
			collections.OrderedDict(sorted(state.order_depths['COCONUT'].sell_orders.items())))
		_, bb_coconuts = self.values_extract(
			collections.OrderedDict(sorted(state.order_depths['COCONUT'].buy_orders.items(), reverse=True)), 1)
		self.coconuts_cache.append((bs_coconuts + bb_coconuts) / 2)
		INF = 1e9
		coconuts_lb = -INF
		coconuts_ub = INF
		if len(self.coconuts_cache) == self.coconuts_dim:
			coconuts_lb = self.calc_next_price_coconuts() - 1
			coconuts_ub = self.calc_next_price_coconuts() + 1
		# Acceptable bid and ask prices
		acc_bid = {'COCONUT' : coconuts_lb }
		acc_ask = {'COCONUT' : coconuts_ub }
		self.steps_coconut += 1

		for person in self.person_position.keys():
			for val in self.person_position[person].keys():
				if person == 'Rhianna':
					self.person_position[person][val] *= 0.5
				if person == "Vladimir":
					self.person_position[person][val] *= 0.5

		for product in ['COCONUT']:
			order_depth: OrderDepth = state.order_depths[product]
			coconut_order = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product])
			orders[product] = coconut_order

		trader_data = self.encodeData(state)
		logger.flush(state, orders, conversions, trader_data)
		return orders, conversions, trader_data