# -*- coding: utf-8 -*-

import asyncio
import os
from random import randint
import sys
from pprint import pprint
import time

# Ruta raíz del proyecto (directorio padre 4 veces)
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root + '/python')

import ccxt.async_support as ccxt # noqa: E402

print('Versión de CCXT:', ccxt.__version__)

####################################################################################
# Bot de arbitraje simple que buscará oportunidades de arbitraje en los mercados ##
# spot y las ejecutará utilizando órdenes de mercado.                                      
#    ##
####################################################################################

# Opciones del bot
tiempo_espera = 5  # segundos para esperar entre cada verificación
trading_ficticio = True  # establecer en falso para ejecutar transacciones reales

# exchanges que desea utilizar para buscar oportunidades de arbitraje
exchanges = [
    ccxt.okx(),
    ccxt.bybit({"options":{"defaultType":"spot"}}),
    ccxt.binance(),
    ccxt.kucoin(),
    ccxt.bitmart(),
    ccxt.gate()
]

# símbolos que desea operar
symbols = [
    "BTC/USDT",
    "LTC/USDT",
    "DOGE/USDT",
    "SHIB/USDT",
    "SOL/USDT",
    "ETH/USDT",
    "ADA/USDT",
    "DOT/USDT",
    "UNI/USDT",
    "LINK/USDT",
]

# tamaños de orden para cada símbolo, ajústelo a su gusto
tamaños_ordenes = {
    "BTC/USDT": 0.001,
    "LTC/USDT": 0.01,
    "DOGE/USDT": 100,
    "SHIB/USDT": 1000000,
    "SOL/USDT": 0.1,
    "ETH/USDT": 0.01,
    "ADA/USDT": 1,
    "DOT/USDT": 0.1,
    "UNI/USDT": 0.1,
    "LINK/USDT": 0.1,
}

async def obtener_ultimos_precios():
    """
    Obtiene los últimos precios de todos los símbolos en todos los exchanges.

    Devuelve:
        Una lista de diccionarios, donde cada diccionario contiene los últimos precios
        de todos los símbolos para un exchange determinado.
    """
    tareas = [exchange.fetch_tickers(symbols) for exchange in exchanges]
    resultados = await asyncio.gather(*tareas)
    return resultados

async def bot():
  """
  Función principal del bot que comprueba si hay oportunidades de arbitraje
  e intenta ejecutarlas.

  """
  prices = await obtener_ultimos_precios()
  for symbol in symbols:
    ms = int(time.time() * 1000)

    # Obtener los precios del símbolo en todos los exchanges
    symbol_prices = [ exchange_prices[symbol]['last'] for exchange_prices in prices ]

    # Encontrar el precio mínimo y máximo del símbolo
    min_price = min(symbol_prices)
    max_price = max(symbol_prices)

    # Obtener el tamaño de la orden para este símbolo
    order_size = tamaños_ordenes[symbol]

    # Encontrar el exchange con el precio mínimo y máximo
    min_exchange = exchanges[symbol_prices.index(min_price)]
    max_exchange = exchanges[symbol_prices.index(max_price)]

    # Calcular la comisión mínima del exchange (taker fee)
    # Advertencia: debe verificar manualmente si hay tarifas de campaña especiales
    min_exchange_fee = min_exchange.fees['trading']['taker']
    min_fee = order_size * min_price * min_exchange_fee

    # Calcular la comisión máxima del exchange (taker fee)
    # Advertencia: debe verificar manualmente si hay tarifas de campaña especiales
    max_exchange_fee = max_exchange.fees['trading']['taker']
    max_fee = order_size * max_price * max_exchange_fee

    # Calcular la ganancia por el diferencial de precios
    price_profit = max_price - min_price

    # Calcular la ganancia neta considerando las comisiones
    profit = (price_profit * order_size) - (min_fee) - (max_fee)

    # Imprimir información si hay oportunidad de arbitraje
    if (profit > 0): # no teniendo en cuenta el slippage o la profundidad del libro de órdenes
      print(ms, symbol, "ganancia:", profit, "Comprar en", min_exchange.id, min_price, "Vender en", max_exchange.id, max_price)
     
      # Ejecutar órdenes solo en modo real (no paper trading)
      if not trading_ficticio:
        buy_min = min_exchange.create_market_buy_order(symbol, order_size)
        sell_max = max_exchange.create_market_sell_order(symbol, order_size)
        orders = await asyncio.gather(buy_min, sell_max) # ejecutarlas "simultáneamente"
        print("Órdenes ejecutadas exitosamente")
    else:
      print(str(ms), symbol, "no hay oportunidad de arbitraje")

async def check_requirements():
  """
  Verifica si los exchanges cumplen con los requisitos para ejecutar el bot.

  - Comprueba si los exchanges soportan la función 'fetchTickers' necesaria para obtener los últimos precios.
  - Verifica si los exchanges soportan los símbolos que se quieren operar.

  En caso de que un exchange no cumpla con alguno de los requisitos, se imprime un mensaje e 
  se finaliza la ejecución del programa con la función 'sys.exit()'.
  """
  
  print("Comprobando si los exchanges soportan fetchTickers y los símbolos que queremos operar")
  for exchange in exchanges:
    if not exchange.has['fetchTickers']:
      print(exchange.id, "no soporta fetchTickers")
      sys.exit()
    await exchange.load_markets()

    for symbol in symbols:
      if symbol not in exchange.markets:
        print(exchange.id, "no soporta", symbol)
        sys.exit()

async def main():

    await check_requirements()

       
    print("Starting bot")

    while True:
        try:
            await bot()
        except e:
            print("Exception: ", e)
        await asyncio.sleep(tiempo_espera)


if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
asyncio.run(main())