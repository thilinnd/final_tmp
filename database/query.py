"""
Database query definitions
"""

# Stock price query
STOCK_PRICE_QUERY = """
SELECT * FROM quote.adjclose
WHERE tickersymbol = %s
AND datetime BETWEEN %s AND %s
ORDER BY datetime
"""

# ETF price query
ETF_PRICE_QUERY = """
SELECT * FROM quote.close
WHERE tickersymbol = %s
AND datetime BETWEEN %s AND %s
ORDER BY datetime
"""

# Futures price query
FUTURES_PRICE_QUERY = """
SELECT c.datetime, c.tickersymbol, c.price
FROM quote.close c
JOIN quote.futurecontractcode fc 
    ON c.datetime = fc.datetime 
    AND fc.tickersymbol = c.tickersymbol
WHERE fc.futurecode = 'VN30F1M'
    AND c.datetime BETWEEN %s AND %s
ORDER BY c.datetime
"""

# VN30 data query
VN30_QUERY = """
SELECT * FROM quote.vn30 v 
WHERE v.datetime >= %s AND v.datetime <= %s
"""

# Daily data query for new DataService
DAILY_DATA_QUERY = """
SELECT EXTRACT(YEAR FROM datetime) as year, datetime as date, tickersymbol, price as close
FROM quote.adjclose
WHERE datetime BETWEEN %s AND %s
ORDER BY datetime, tickersymbol
"""

# Financial info query for new DataService
FINANCIAL_INFO_QUERY = """
SELECT year, tickersymbol, value, code
FROM financial.financial_info
WHERE year BETWEEN %s AND %s
AND tickersymbol IN %s
ORDER BY year, tickersymbol
"""

# Index data query for new DataService
INDEX_QUERY = """
SELECT date, open, close
FROM quote.index
WHERE date BETWEEN %s AND %s
ORDER BY date
"""

# VN30F1 matched data query
MATCHED_QUERY = """
  select m.datetime, m.tickersymbol, m.price
  from quote.matched m
  join quote.futurecontractcode fc on date(m.datetime) = fc.datetime and fc.tickersymbol = m.tickersymbol
  where fc.futurecode = %s and m.datetime between %s and %s and
        ((EXTRACT(HOUR FROM m.datetime) >= 9 AND EXTRACT(HOUR FROM m.datetime) < 14)
        OR (EXTRACT(HOUR FROM m.datetime) = 14 AND EXTRACT(MINUTE FROM m.datetime) <= 30))
  order by m.datetime
"""

# VN30F1 bid ask data query
BID_ASK_QUERY = """
  select b.datetime, b.tickersymbol, b.price, a.price, a.price - b.price
  from quote.bidprice b join quote.askprice a
  on b.datetime = a.datetime and b.tickersymbol = a.tickersymbol and b.depth = a.depth
  join quote.futurecontractcode fc on date(b.datetime) = fc.datetime and fc.tickersymbol = b.tickersymbol
  where b.depth = 1 and fc.futurecode = %s and b.datetime between %s and %s and
        ((EXTRACT(HOUR FROM b.datetime) >= 9 AND EXTRACT(HOUR FROM b.datetime) < 14)
        OR (EXTRACT(HOUR FROM b.datetime) = 14 AND EXTRACT(MINUTE FROM b.datetime) <= 30))
  order by b.datetime
"""

# VN30F1 close price query
CLOSE_QUERY = """
  select c.datetime, c.tickersymbol, c.price
  from quote.close c
  join quote.futurecontractcode fc on c.datetime = fc.datetime and fc.tickersymbol = c.tickersymbol
  where fc.futurecode = %s and c.datetime between %s and %s
  order by c.datetime
"""