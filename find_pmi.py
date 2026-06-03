from fredapi import Fred
fred = Fred(api_key="9ac5ce275222ddff1b3b91180f76d048")
print(fred.get_series('NAPM').tail(5))
