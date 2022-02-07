- [Guidance: Analysis](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_analysis.md)
- [Guidance: Dataset](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_dataset.md)
- [Guidance: Model](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_model.md)
- [Guidance: Management](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_management.md)
- [Guidance: StockValuation](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_stock_valuation.md)

---

## Framework supporting financial dataset
### FianaceDataReader
`fundamentals`
```python
import FinanceDataReader as fdr
fdr.StockListing('KRX-MARCAP')
```
`market indicies`
```python
import FinanceDataReader as fdr
fdr.StockListing('KRX')
fdr.StockListing('KOSPI')
fdr.StockListing('KOSDAQ')
fdr.StockListing('KONEX')
fdr.StockListing('NYSE')
fdr.StockListing('NASDAQ')
fdr.StockListing('AMEX')
fdr.StockListing('S&P500')
fdr.StockListing('SSE')
fdr.StockListing('SZSE')
fdr.StockListing('HKEX')
fdr.StockListing('TSE')
fdr.StockListing('HOSE')
```

- **FUTURE**: 'NG', 'GC', 'SI', 'HG', 'CL'
- **MARKET**: 'KS11', 'KQ11', 'KS50', 'KS100', 'KRX100', 'KS200', 'DJI', 'IXIC', 'US500', 'RUTNU', 'VIX', 'JP225', 'STOXX50', 'HK50', 'CSI300', 'TWII', 'HNX30', 'SSEC', 'UK100', 'DE30', 'FCHI'
- **EXCHANGE RATE**: 'USD/KRW', 'USD/EUR', 'USD/JPY', 'CNY/KRW', 'EUR/USD', 'USD/JPY', 'JPY/KRW', 'AUD/USD', 'EUR/JPY', 'USD/RUB'
- **GOVERNMENT BOND**: 'KR1YT=RR', 'KR2YT=RR', 'KR3YT=RR', 'KR4YT=RR', 'KR5YT=RR', 'KR10YT=RR', 'KR20YT=RR', 'KR30YT=RR', 'KR50YT=RR', 'US1MT=X', 'US3MT=X', 'US6MT=X', 'US1YT=X', 'US2YT=X', 'US3YT=X', 'US5YT=X', 'US7YT=X','US10YT=X', 'US30YT=X'
- **CRYPTOCURRENCY**: 'BTC/KRW','ETH/KRW','XRP/KRW','BCH/KRW','EOS/KRW','LTC/KRW','XLM/KRW', 'BTC/USD','ETH/USD','XRP/USD','BCH/USD','EOS/USD','LTC/USD','XLM/USD'

```python
import FinanceDataReader as fdr

fdr.DataReader('NG')
fdr.DataReader('GC')
fdr.DataReader('SI')
fdr.DataReader('HG')
fdr.DataReader('CL')
fdr.DataReader('KS11')
fdr.DataReader('KQ11')
fdr.DataReader('KS50')
fdr.DataReader('KS100')
fdr.DataReader('KS200')
fdr.DataReader('DJI')
fdr.DataReader('IXIC')
fdr.DataReader('US500')
fdr.DataReader('RUTNU')
fdr.DataReader('VIX')
fdr.DataReader('JP225')
fdr.DataReader('STOXX50')
fdr.DataReader('HK50')
fdr.DataReader('CSI300')
fdr.DataReader('TWII')
fdr.DataReader('HNX30')
fdr.DataReader('SSEC')
fdr.DataReader('UK100')
fdr.DataReader('DE30')
fdr.DataReader('FCHI')
fdr.DataReader('USD/KRW')
fdr.DataReader('USD/EUR')
fdr.DataReader('USD/JPY')
fdr.DataReader('CNY/KRW')
fdr.DataReader('EUR/USD')
fdr.DataReader('JPY/KRW')
fdr.DataReader('AUD/USD')
fdr.DataReader('EUR/JPY')
fdr.DataReader('USD/RUB')
fdr.DataReader('KR1YT=RR')
fdr.DataReader('KR2YT=RR')
fdr.DataReader('KR3YT=RR')
fdr.DataReader('KR4YT=RR')
fdr.DataReader('KR5YT=RR')
fdr.DataReader('KR10YT=RR')
fdr.DataReader('KR20YT=RR')
fdr.DataReader('KR30YT=RR')
fdr.DataReader('KR50YT=RR')
fdr.DataReader('US1MT=X')
fdr.DataReader('US3MT=X')
fdr.DataReader('US6MT=X')
fdr.DataReader('US1YT=X')
fdr.DataReader('US2YT=X')
fdr.DataReader('US3YT=X')
fdr.DataReader('US5YT=X')
fdr.DataReader('US7YT=X')
fdr.DataReader('US10YT=X')
fdr.DataReader('US30YT=X')
fdr.DataReader('BTC/KRW')
fdr.DataReader('ETH/KRW')
fdr.DataReader('XRP/KRW')
fdr.DataReader('BCH/KRW')
fdr.DataReader('EOS/KRW')
fdr.DataReader('LTC/KRW')
fdr.DataReader('XLM/KRW')
fdr.DataReader('BTC/USD')
fdr.DataReader('ETH/USD')
fdr.DataReader('XRP/USD')
fdr.DataReader('BCH/USD')
fdr.DataReader('EOS/USD')
fdr.DataReader('LTC/USD')
fdr.DataReader('XLM/USD')
```
`tickers`
```python
import FinanceDataReader as fdr
fdr.DataReader('005930')
```

<br><br><br>

---


### Yahooquery

`ticker module summary`
```python
from yahooquery import Ticker

ticker = Ticker('ARE')
ticker.summary_detail
ticker.calendar_events
ticker.company_officers
ticker.earning_history
ticker.earnings
ticker.earnings_trend
ticker.esg_scores
ticker.financial_data
ticker.fund_bond_holdings
ticker.fund_bond_holdings
ticker.fund_bond_ratings
ticker.fund_equity_holdings
ticker.fund_holding_info
ticker.fund_ownership
ticker.fund_performance
ticker.fund_profile
ticker.fund_sector_weightings
ticker.fund_top_holdings
ticker.grading_history
ticker.index_trend
ticker.industry_trend
ticker.insider_holders
ticker.insider_transactions
ticker.institution_ownership
ticker.key_stats
ticker.major_holders
ticker.page_views
ticker.price
ticker.quote_type
ticker.recommendation_trend
ticker.sec_filings
ticker.share_purchase_activity
ticker.summary_detail
ticker.summary_profile
```
`ticker multiple modules`
```python
from yahooquery import Ticker

ticker = Ticker('ARE')
ticker.all_modules

# modules: assetProfile, balanceSheetHistory, balanceSheetHistoryQuarterly, calendarEvents, cashflowStatementHistory, cashflowStatementHistoryQuarterly, defaultKeyStatistics, earnings, earningsHistory, earningsTrend, esgScores, financialData, fundOwnership, fundPerformance, fundProfile, indexTrend, incomeStatementHistory, incomeStatementHistoryQuarterly, industryTrend, insiderHolders, insiderTransactions, institutionOwnership, majorHoldersBreakdown, pageViews, price, quoteType, recommendationTrend, secFilings, netSharePurchaseActivity, sectorTrend, summaryDetail, summaryProfile, topHoldings, upgradeDowngradeHistory
ticker.get_modules(modules='balanceSheetHistory')
```

#### [Ticker] Historical Prices
`ticker.history`
```python
from yahooquery import Ticker

# period: '1d', '5d', '7d', '60d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
# interval: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
ticker = Ticker('ARE', asynchronous=True)
ticker.history(period='1mo', interval='1m', start=None, end=None)
```

#### [Ticker] Financials
`individual`
```python
from yahooquery import Ticker

ticker = Ticker('ARE')
ticker.balance_sheet(frequency='a', trailing=True) # Annual: a, Quarter: q
ticker.cash_flow(frequency='a', trailing=True)  # Annual: a, Quarter: q
ticker.income_statement(frequency='a', trailing=True)  # Annual: a, Quarter: q
ticker.valuation_measures
```

`multiple`
```python
from yahooquery import Ticker

ticker = Ticker('ARE')
ticker.all_financial_data()

types = ['TotalDebt', 'TotalAssets', 'EBIT', 'EBITDA', 'PeRatio']
ticker.get_financial_data(types, trailing=False)
""" types
# Balance Sheet
    'AccountsPayable', 'AccountsReceivable', 'AccruedInterestReceivable',
    'AccumulatedDepreciation', 'AdditionalPaidInCapital',
    'AllowanceForDoubtfulAccountsReceivable', 'AssetsHeldForSaleCurrent',
    'AvailableForSaleSecurities', 'BuildingsAndImprovements', 'CapitalLeaseObligations',
    'CapitalStock', 'CashAndCashEquivalents', 'CashCashEquivalentsAndShortTermInvestments',
    'CashEquivalents', 'CashFinancial', 'CommercialPaper', 'CommonStock',
    'CommonStockEquity', 'ConstructionInProgress', 'CurrentAccruedExpenses',
    'CurrentAssets', 'CurrentCapitalLeaseObligation', 'CurrentDebt',
    'CurrentDebtAndCapitalLeaseObligation', 'CurrentDeferredAssets',
    'CurrentDeferredLiabilities', 'CurrentDeferredRevenue', 'CurrentDeferredTaxesAssets',
    'CurrentDeferredTaxesLiabilities', 'CurrentLiabilities', 'CurrentNotesPayable',
    'CurrentProvisions', 'DefinedPensionBenefit', 'DerivativeProductLiabilities',
    'DividendsPayable', 'DuefromRelatedPartiesCurrent', 'DuefromRelatedPartiesNonCurrent',
    'DuetoRelatedPartiesCurrent', 'DuetoRelatedPartiesNonCurrent', 'EmployeeBenefits',
    'FinancialAssets', 'FinancialAssetsDesignatedasFairValueThroughProfitorLossTotal',
    'FinishedGoods', 'FixedAssetsRevaluationReserve', 'ForeignCurrencyTranslationAdjustments',
    'GainsLossesNotAffectingRetainedEarnings', 'GeneralPartnershipCapital', 'Goodwill',
    'GoodwillAndOtherIntangibleAssets', 'GrossAccountsReceivable', 'GrossPPE',
    'HedgingAssetsCurrent', 'HeldToMaturitySecurities', 'IncomeTaxPayable',
    'InterestPayable', 'InventoriesAdjustmentsAllowances', 'Inventory',
    'InvestedCapital', 'InvestmentProperties', 'InvestmentinFinancialAssets',
    'InvestmentsAndAdvances', 'InvestmentsInOtherVenturesUnderEquityMethod',
    'InvestmentsinAssociatesatCost', 'InvestmentsinJointVenturesatCost',
    'InvestmentsinSubsidiariesatCost', 'LandAndImprovements', 'Leases',
    'LiabilitiesHeldforSaleNonCurrent', 'LimitedPartnershipCapital',
    'LineOfCredit', 'LoansReceivable', 'LongTermCapitalLeaseObligation',
    'LongTermDebt', 'LongTermDebtAndCapitalLeaseObligation', 'LongTermEquityInvestment',
    'LongTermProvisions', 'MachineryFurnitureEquipment', 'MinimumPensionLiabilities',
    'MinorityInterest', 'NetDebt', 'NetPPE', 'NetTangibleAssets', 'NonCurrentAccountsReceivable',
    'NonCurrentAccruedExpenses', 'NonCurrentDeferredAssets', 'NonCurrentDeferredLiabilities',
    'NonCurrentDeferredRevenue', 'NonCurrentDeferredTaxesAssets', 'NonCurrentDeferredTaxesLiabilities',
    'NonCurrentNoteReceivables', 'NonCurrentPensionAndOtherPostretirementBenefitPlans',
    'NonCurrentPrepaidAssets', 'NotesReceivable', 'OrdinarySharesNumber',
    'OtherCapitalStock', 'OtherCurrentAssets', 'OtherCurrentBorrowings',
    'OtherCurrentLiabilities', 'OtherEquityAdjustments', 'OtherEquityInterest',
    'OtherIntangibleAssets', 'OtherInventories', 'OtherInvestments', 'OtherNonCurrentAssets',
    'OtherNonCurrentLiabilities', 'OtherPayable', 'OtherProperties', 'OtherReceivables',
    'OtherShortTermInvestments', 'Payables', 'PayablesAndAccruedExpenses',
    'PensionandOtherPostRetirementBenefitPlansCurrent', 'PreferredSecuritiesOutsideStockEquity',
    'PreferredSharesNumber', 'PreferredStock', 'PreferredStockEquity',
    'PrepaidAssets', 'Properties', 'RawMaterials', 'Receivables',
    'ReceivablesAdjustmentsAllowances', 'RestrictedCash', 'RestrictedCommonStock',
    'RetainedEarnings', 'ShareIssued', 'StockholdersEquity', 'TangibleBookValue',
    'TaxesReceivable', 'TotalAssets', 'TotalCapitalization', 'TotalDebt',
    'TotalEquityGrossMinorityInterest', 'TotalLiabilitiesNetMinorityInterest',
    'TotalNonCurrentAssets', 'TotalNonCurrentLiabilitiesNetMinorityInterest',
    'TotalPartnershipCapital', 'TotalTaxPayable', 'TradeandOtherPayablesNonCurrent',
    'TradingSecurities', 'TreasurySharesNumber', 'TreasuryStock', 'UnrealizedGainLoss',
    'WorkInProcess', 'WorkingCapital'

# Cash Flow
    'RepaymentOfDebt', 'RepurchaseOfCapitalStock', 'CashDividendsPaid',
    'CommonStockIssuance', 'ChangeInWorkingCapital',
    'CapitalExpenditure',
    'CashFlowFromContinuingFinancingActivities', 'NetIncome',
    'FreeCashFlow', 'ChangeInCashSupplementalAsReported',
    'SaleOfInvestment', 'EndCashPosition', 'OperatingCashFlow',
    'DeferredIncomeTax', 'NetOtherInvestingChanges',
    'ChangeInAccountPayable', 'NetOtherFinancingCharges',
    'PurchaseOfInvestment', 'ChangeInInventory',
    'DepreciationAndAmortization', 'PurchaseOfBusiness',
    'InvestingCashFlow', 'ChangesInAccountReceivables',
    'StockBasedCompensation', 'OtherNonCashItems',
    'BeginningCashPosition'

# Income Satetment
    'Amortization', 'AmortizationOfIntangiblesIncomeStatement',
    'AverageDilutionEarnings', 'BasicAccountingChange', 'BasicAverageShares',
    'BasicContinuousOperations', 'BasicDiscontinuousOperations', 'BasicEPS',
    'BasicEPSOtherGainsLosses', 'BasicExtraordinary', 'ContinuingAndDiscontinuedBasicEPS',
    'ContinuingAndDiscontinuedDilutedEPS', 'CostOfRevenue', 'DepletionIncomeStatement',
    'DepreciationAmortizationDepletionIncomeStatement', 'DepreciationAndAmortizationInIncomeStatement',
    'DepreciationIncomeStatement', 'DilutedAccountingChange', 'DilutedAverageShares',
    'DilutedContinuousOperations', 'DilutedDiscontinuousOperations', 'DilutedEPS',
    'DilutedEPSOtherGainsLosses', 'DilutedExtraordinary', 'DilutedNIAvailtoComStockholders',
    'DividendPerShare', 'EBIT', 'EBITDA', 'EarningsFromEquityInterest',
    'EarningsFromEquityInterestNetOfTax', 'ExciseTaxes', 'GainOnSaleOfBusiness',
    'GainOnSaleOfPPE', 'GainOnSaleOfSecurity', 'GeneralAndAdministrativeExpense',
    'GrossProfit', 'ImpairmentOfCapitalAssets', 'InsuranceAndClaims',
    'InterestExpense', 'InterestExpenseNonOperating', 'InterestIncome',
    'InterestIncomeNonOperating', 'MinorityInterests', 'NetIncome', 'NetIncomeCommonStockholders',
    'NetIncomeContinuousOperations', 'NetIncomeDiscontinuousOperations',
    'NetIncomeExtraordinary', 'NetIncomeFromContinuingAndDiscontinuedOperation',
    'NetIncomeFromContinuingOperationNetMinorityInterest', 'NetIncomeFromTaxLossCarryforward',
    'NetIncomeIncludingNoncontrollingInterests', 'NetInterestIncome',
    'NetNonOperatingInterestIncomeExpense', 'NormalizedBasicEPS', 'NormalizedDilutedEPS',
    'NormalizedEBITDA', 'NormalizedIncome', 'OperatingExpense', 'OperatingIncome',
    'OperatingRevenue', 'OtherGandA', 'OtherIncomeExpense', 'OtherNonOperatingIncomeExpenses',
    'OtherOperatingExpenses', 'OtherSpecialCharges', 'OtherTaxes',
    'OtherunderPreferredStockDividend', 'PreferredStockDividends',
    'PretaxIncome', 'ProvisionForDoubtfulAccounts', 'ReconciledCostOfRevenue',
    'ReconciledDepreciation', 'RentAndLandingFees', 'RentExpenseSupplemental',
    'ReportedNormalizedBasicEPS', 'ReportedNormalizedDilutedEPS', 'ResearchAndDevelopment',
    'RestructuringAndMergernAcquisition', 'SalariesAndWages', 'SecuritiesAmortization',
    'SellingAndMarketingExpense', 'SellingGeneralAndAdministration', 'SpecialIncomeCharges',
    'TaxEffectOfUnusualItems', 'TaxLossCarryforwardBasicEPS', 'TaxLossCarryforwardDilutedEPS',
    'TaxProvision', 'TaxRateForCalcs', 'TotalExpenses', 'TotalOperatingIncomeAsReported',
    'TotalOtherFinanceCost', 'TotalRevenue', 'TotalUnusualItems',
    'TotalUnusualItemsExcludingGoodwill', 'WriteOff'

# Valuation Measures
    'ForwardPeRatio', 'PsRatio', 'PbRatio',
    'EnterprisesValueEBITDARatio', 'EnterprisesValueRevenueRatio',
    'PeRatio', 'MarketCap', 'EnterpriseValue', 'PegRatio'
"""
```

#### [Ticker] Options
```python
from yahooquery import Ticker

ticker = Ticker('ARE')
ticker.option_chain
```

#### [Ticker] Miscellaneous
```python
from yahooquery import Ticker

tickers = Ticker(['ARE', 'AAPL', 'GL'])
tickers.corporate_events
```
```python
from yahooquery import Ticker

tickers = Ticker(['ARE', 'AAPL', 'GL'])
tickers.news(5)
```
```python
from yahooquery import Ticker

tickers = Ticker(['ARE', 'AAPL', 'GL'])
tickers.quotes
```
```python
from yahooquery import Ticker

tickers = Ticker(['ARE', 'AAPL', 'GL'])
tickers.recommendations
```
```python
from yahooquery import Ticker

tickers = Ticker(['ARE', 'AAPL', 'GL'])
tickers.technical_insights
```


#### [Ticker] Module Summary
`ticker.summary_detail`
```python
from yahooquery import Ticker
import pandas as pd

ticker = Ticker('ARE')
summary = pd.DataFrame(ticker.summary_detail)
summary # funcdamentials
```

`ticker.asset_profile`
```python
import pandas as pd
from yahooquery import Ticker

# Retrieve each company's profile information
ticker = Ticker('ARE')
profile = pd.DataFrame(ticker.asset_profile['ARE'])
for idx, company_officer_dict in enumerate(profile['companyOfficers']):
    company_officer = pd.DataFrame({k:[v] for k, v in company_officer_dict.items()}) if not idx else company_officer.append(pd.DataFrame({k:[v] for k, v in company_officer_dict.items()}))
profile = pd.concat([profile, company_officer.reset_index(drop=True)], axis=1)
profile
```



<br><br><br>

---


### Pandas DataReader

```python
from pandas_datareader import data

test = data.DataReader(['TSLA', 'FB'], 'yahoo', start='2018/01/01', end='2019/12/31')
test.head()
```

<br><br><br>

---

### Crawl: Finviz
```python
import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get('https://finviz.com/quote.ashx?t=AAPL', headers=headers)
soup = BeautifulSoup(response.text)
target_table_tag = soup.find('table', attrs={'class': 'snapshot-table2'}) #tables = soup.find_all('table')
df = pd.read_html(str(target_table_tag))[0]

df.columns = ['key', 'value'] * 6
df_list = [df.iloc[:, i*2: i*2+2] for i in range(6)]
df_factor = pd.concat(df_list, ignore_index=True)
df_factor.set_index('key', inplace=True)
df_factor
```

<br><br><br>

---


## Initialization
```python
import pandas as pd
from ailever.investment import market_information, Loader

loader = Loader()
loader.into_local()

df = market_information(market_cap=False)
Df = loader.from_local(baskets=df[df.Market=='KOSPI'].Symbol.to_list(), mode='Close')
pd.DataFrame(data=Df[0], columns=Df[1].Name.to_list())
```


## Market Information
```python
from ailever.investment import market_information
df = market_information(baskets=None, only_symbol=False, inverse_mapping=False, market_cap=False)
df[0]
```

```python
from ailever.investment import market_information
df = market_information(baskets=['삼성전자', 'SK하이닉스'], only_symbol=True, inverse_mapping=False, market_cap=False)
df
```

```python
from ailever.investment import market_information
df = market_information(baskets=['005930', '000660'], only_symbol=False, inverse_mapping=True, market_cap=False)
df
```

## Sector
```python
from ailever.investment import sectors

tickers = sectors.us_reit()
tickers.list
tickers.pdframe
tickers.subsector

tickers = sectors.us_reit(subsector='Office')
tickers.list
tickers.pdframe
```

## Data Vendor
```python
```

## Parallelizer
```python
from ailever.investment import prllz_loader

datacore = prllz_loader(baskets=['ARE', 'BXP', 'O'])
datacore.ndarray
datacore.pdframe
```

```python
from ailever.investment import market_information
from ailever.investment import parallelize
import pandas as pd

df = market_information()
baskets = df.loc[lambda x: x.Market == 'KOSPI'].dropna().reset_index().drop('index', axis=1).Symbol.to_list()
sample_columns = pd.read_csv('.fmlops/feature_store/1d/005390.csv').columns.to_list()

DTC = parallelize(baskets=baskets, path='.fmlops/feature_store/1d', base_column='Close', date_column='Date', columns=sample_columns)
DTC.pdframe
```

## Integrated Loader
```python
from ailever.investment import market_information
from ailever.investment import Loader 

df = market_information().dropna()
df[df.Industry.str.contains('리츠')].loc[lambda x:x.Market == 'NYSE']

loader = Loader()
dataset = loader.ohlcv_loader(baskets=['ARE', 'O', 'BXP'])
dataset.dict
dataset.log

modules = loader.fmf  # '--> modules search for fundmentals'
modules = loader.fundamentals_modules_fromyahooquery

dataset = loader.fundamentals_loader(baskets=['ARE', 'O', 'BXP'], sort_by='Marketcap')
dataset.dict
dataset.log
```


```python
from ailever.investment import Loader
loader = Loader()
loader.into_local()
```

## Preprocessor
```python
from ailever.investment import Preprocessor

pre = Preprocessor() #'''for ticker processing'''
pre.pct_change(baskets=['ARE','O','BXP'], window=[1,3,5],kind='ticker') #'''for index preprocessed data attachment'''
pre.pct_change(baskets=['^VIX'], kind='index_full') #'''including index ohlcv'''
pre.pct_change(baskets=['^VIX'], kind='index_single') #'''Only preprocessed index data

pre.overnight(baskets=['ARE','O','BXP'], kind='index_full') #'''including index ohlcv
pre.rolling(baskets=['ARE','O','BXP'], kind='index_full') #'''including index ohlcv

pre.date_featuring()
pre.na_handler()

pre.preprocess_list
pre.to_csv(option='dropna')
pre.reset()
```

