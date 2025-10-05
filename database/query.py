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
