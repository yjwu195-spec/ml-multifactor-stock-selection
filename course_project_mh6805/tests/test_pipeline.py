import unittest
import numpy as np
import pandas as pd
from src.data import financial_alignment
from src.modeling import TrainWinsorizer, split_panel
from src.backtest import metrics, simulate


class TemporalAndPortfolioChecks(unittest.TestCase):
    def test_future_filing_is_unavailable(self):
        left = pd.DataFrame({'ts_code':['A','A'], 'signal_date':pd.to_datetime(['20200105','20200205'])})
        right = pd.DataFrame({'ts_code':['A','A'], 'available_date':pd.to_datetime(['20200102','20200202']), 'report_end':pd.to_datetime(['20190930','20191231']), 'roe':[1.,999.]})
        actual = financial_alignment(left,right,['roe'],550)
        self.assertEqual(actual.roe.tolist(),[1.,999.])

    def test_old_period_late_revision_does_not_override_new_filing(self):
        left = pd.DataFrame({'ts_code':['A'], 'signal_date':pd.to_datetime(['20200501'])})
        right = pd.DataFrame({'ts_code':['A','A'], 'available_date':pd.to_datetime(['20200401','20200415']), 'report_end':pd.to_datetime(['20191231','20190930']), 'roe':[2.,999.]})
        self.assertEqual(financial_alignment(left,right,['roe'],550).roe.iloc[0],2.)

    def test_stale_financial_values_stay_missing(self):
        left=pd.DataFrame({'ts_code':['A'],'signal_date':pd.to_datetime(['20220101'])})
        right=pd.DataFrame({'ts_code':['A'],'available_date':pd.to_datetime(['20200101']),'report_end':pd.to_datetime(['20190930']),'roe':[9.]})
        self.assertTrue(financial_alignment(left,right,['roe'],550).roe.isna().all())

    def test_train_only_winsorization(self):
        fitted=TrainWinsorizer().fit([[0.],[1.],[2.]])
        before=fitted.upper_.copy()
        fitted.transform([[1e12]])
        np.testing.assert_array_equal(before,fitted.upper_)

    def test_initial_loss_enters_drawdown(self):
        self.assertAlmostEqual(metrics([-.2,.1])['max_drawdown'],.2)

    def test_compound_return(self):
        self.assertAlmostEqual(metrics([.1,-.1])['total_return'],-.01)

    def arrays(self):
        dates=pd.date_range('20200102',periods=4)
        return {'dates':dates,'codes':pd.Index(['A','B']), 'mark':np.array([[10.,10.],[11.,10.],[12.,10.],[12.,10.]]), 'tradable':np.ones((4,2),dtype=bool),'forced_delist':np.zeros((4,2),dtype=bool),'benchmark':np.ones(4)}

    def test_labels_do_not_change_selected_holdings(self):
        arrays=self.arrays()
        scored=pd.DataFrame({'execution_date':[arrays['dates'][0]]*2,'exit_date':[pd.NaT]*2,'score':[.9,.1],'ts_code':['A','B'],'label':[0.,1.]})
        first=simulate(scored,arrays,0,{'top_n':1})
        scored['label']=[1.,0.]
        second=simulate(scored,arrays,0,{'top_n':1})
        np.testing.assert_array_equal(first.nav,second.nav)
        self.assertAlmostEqual(first.nav.iloc[-1],1.2)

    def test_unavailable_entry_leaves_cash(self):
        arrays=self.arrays()
        arrays['tradable'][0,0]=False
        scored=pd.DataFrame({'execution_date':[arrays['dates'][0]],'exit_date':[pd.NaT],'score':[.9],'ts_code':['A']})
        actual=simulate(scored,arrays,10,{'top_n':1})
        self.assertEqual(actual.nav.iloc[-1],1.)
        self.assertEqual(actual.blocked_new_orders.sum(),1)

    def test_transaction_cost_does_not_create_cash(self):
        arrays=self.arrays()
        scored=pd.DataFrame({'execution_date':[arrays['dates'][0]],'exit_date':[pd.NaT],'score':[.9],'ts_code':['A']})
        free=simulate(scored,arrays,0,{'top_n':1})
        paid=simulate(scored,arrays,10,{'top_n':1})
        self.assertLess(paid.nav.iloc[-1],free.nav.iloc[-1])
        self.assertTrue((paid.cash_weight >= -1e-8).all())


if __name__=='__main__':
    unittest.main()
