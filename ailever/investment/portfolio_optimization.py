from .fmlops_loader_system import parallelize
from .mathematical_modules import regressor, scaler

class PortfolioManagement:
    def __init__(self, baskets, period=100):
        self.pdframe = parallelize(baskets=baskets, period=period).pdframe
        self.initialization()

    def initialization(self):
        self.hiest_momentum_stocks = _evaludate_momentum()
        self.return_matrix = None
        
    def _evaludate_momentum(self, Df=None, ADf=None, filter_period=300, capital_priority=True, regressor_criterion=1.5, seasonal_criterion=0.3, GC=False, V='KS11', download=False):
        assert bool(Df or ADf), 'Dataset Df or ADf must be defined.'
        self.dummies = dummies()
        self.dummies.__init__ = dict()

        # .Log folder
        if not os.path.isdir('.Log') : os.mkdir('.Log')

        # Korean Font Set
        font_names = list()
        for font in fm.fontManager.ttflist:
            font_names.append(font.name)
        self.dummies.__init__['fonts'] = font_names
        if 'NanumBarunGothic' in font_names:
            plt.rcParams["font.family"] = 'NanumBarunGothic'
        
        # Model based on Deep Learning(Neural Network)
        self.deepNN = None

        if ADf:
            self.ADf = ADf
            self.Df = self.ADf['Close']
        else:
            self.Df = Df
        
        if capital_priority:
            norm = scaler.standard(self.Df[0][-filter_period:])
        else:
            norm = scaler.minmax(self.Df[0][-filter_period:])

        yhat = regressor(norm)
        container = yhat[-1,:] - yhat[0,:]

        self.index = list()
        self._index0 = np.where(container>=regressor_criterion)[0]

        recommended_stock_info = self.Df[1].iloc[self._index0]
        alert = list(zip(recommended_stock_info.Name.tolist(), recommended_stock_info.Symbol.tolist())); print(alert)
        

        # Short Term Investment Stock
        long_period = 300
        short_period = 30
        back_shifting = 0
        print('\n* Short Term Trade List')
        for i in self._index0:
            info = (i, long_period, short_period, back_shifting)
            selected_stock_info = self.Df[1].iloc[info[0]]
            result = self._stock_decompose(info[0], info[1], info[2], info[3], decompose_type='stl', resid_transform=True)

            x = scaler.minmax(result.seasonal)
            index = {}
            index['ref'] = set([295,296,297,298,299])
            index['min'] = set(np.where((x<seasonal_criterion) & (x>=0))[0])
            if index['ref']&index['min']:
                self.index.append(info[0])
                print(f'  - {selected_stock_info.Name}({selected_stock_info.Symbol}) : {info[0]}')

        if GC:
            self.Granger_C()

        # Visualization
        if V:
            df = pd.DataFrame(self.Df[0][:, self.index])
            df.columns = self.Df[1].iloc[self.index].Name
            ks11 = Df[3][V][self.Df[4]][-len(df):].reset_index().drop('index', axis=1)
            ks11.columns = [V]
            df = pd.concat([ks11, df], axis=1)

            plt.figure(figsize=(13,25)); layout = (5,1); axes = dict()
            axes[0] = plt.subplot2grid(layout, (0, 0), rowspan=1)
            axes[1] = plt.subplot2grid(layout, (1, 0), rowspan=1)
            axes[2] = plt.subplot2grid(layout, (2, 0), rowspan=1)
            axes[3] = plt.subplot2grid(layout, (3, 0), rowspan=2)

            for name, stock in zip(df.columns, df.values.T):
                axes[0].plot(stock, label=name)
                axes[0].text(len(stock), stock[-1], name)
            axes[0].set_title('STOCK')
            axes[0].legend(loc='lower left')
            axes[0].grid(True)

            df.diff().plot(ax=axes[1])
            axes[1].set_title('DIFF')
            axes[1].legend(loc='lower left')
            axes[1].grid(True)

            for i, name in enumerate(df.columns):
                pd.plotting.autocorrelation_plot(df.diff().dropna().iloc[:,i], ax=axes[2], label=name)
            axes[2].set_title('ACF')
            axes[2].grid(True)
            axes[2].legend(loc='upper right')
            
            # Correlation
            axes[3].set_title('Correlation')
            sns.set_theme(style="white")
            sns.despine(left=True, bottom=True)
            mask = np.triu(np.ones_like(df.corr(), dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(df.corr(), mask=mask, cmap=cmap, square=True, annot=True, linewidths=.5, ax=axes[3])
            
            plt.tight_layout()
            if download:
                plt.savefig('Ailf_KR.pdf')
            #plt.show()

    def risks(self):
        self.risk_matrix = None

    def portfolio_selection(self):
        # TODO : from MCDA
        pass

    def portfolio_optimization(self):
        # TODO : deepdow-like
        pass


class MultiCriteriaDecisionAnalysis:
    def TOPSIS(self):
        # Technique for Order of Preference by Similarity to Ideal Solution
        pass

    def MAUT(self):
        # Multi-Attribute Utility Thoery
        utility_score = None
        return utility_score

    def ELECTRE(self):
        # ELimination Et Choice Translating REality
        pass

    def PROMETHEE(self):
        # Preference ranking organization method for enrichment evaluation
        pass


