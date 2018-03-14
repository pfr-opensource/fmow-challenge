import numpy as np
import pandas as pd

from fmow_helper import (
        BASELINE_CATEGORIES, MIN_WIDTHS, WIDTHS, centrality, softmax, lerp, create_submission,
        csv_parse, read_merged_Plog
        )

BASELINE_CNN_NM = 'baseline/data/output/predictions/soft-predictions-cnn-no_metadata.txt'
BASELINE_CNN = 'baseline/data/output/predictions/soft-predictions-cnn-with_metadata.txt'
BASELINE_LSTM = 'baseline/data/output/predictions/soft-predictions-lstm-with_metadata.txt'

def P_baseline():
    """
    Baseline predicted probabilities, ensembled from:
    - CNN, no metadata
    - CNN, with metadata
    - LSTM, with metadata
    """
    nP_nm_cnn = pd.read_csv(BASELINE_CNN_NM, names=BASELINE_CATEGORIES, index_col=0).sort_index()
    nP_cnn = pd.read_csv(BASELINE_CNN, names=BASELINE_CATEGORIES, index_col=0).sort_index()
    P_lstm = pd.read_csv(BASELINE_LSTM, names=BASELINE_CATEGORIES, index_col=0).sort_index()
    P_cnn = nP_cnn.div(nP_cnn.sum(1).round(), 0)
    P_nm_cnn = nP_nm_cnn.div(nP_nm_cnn.sum(1).round(), 0)
    P_m_test = lerp(0.56, P_cnn, P_lstm)
    P_test = lerp(0.07, P_m_test, P_nm_cnn)
    return P_test

def P_no_baseline():
    """
    Predicted probabilities before ensembling with baseline.
    """
    test = csv_parse('working/metadata/boxes-test-rgb.csv')
    Plog_test = read_merged_Plog()

    Plog = Plog_test.groupby(test.ID).mean()
    df = test.groupby('ID').first()

    # The prediction above doesn't use any image metadata.
    # We remedy that by applying basic priors about the dataset.
    assert Plog.index.isin(df.index).all()
    assert df.width_m.isin([500, 1500, 5000]).all()
    Plog = Plog.apply(lambda ser:
                      ser.where(df.width_m >= MIN_WIDTHS[ser.name], -np.inf) - 1.2 * ~df.width_m.loc[ser.index].isin(WIDTHS[ser.name])
                      if ser.name!='false_detection' else ser)

    df2 = df.loc[Plog.index]
    r = centrality(df2)
    Plog['false_detection'] += (.5 + .7 * (df2.width_m==500)) * (2. * (r>=.3) - .5) - 1

    return softmax(Plog)

def P_ensemble():
    """
    Predicted probabilities for each class.
    """
    eps = 1e-6
    Plog_mix = lerp(0.71, np.log(P_baseline()+eps), np.log(P_no_baseline()+eps))
    Plog_mix['false_detection'] -= 0.43
    P_mix = softmax(Plog_mix)
    P_mix['flooded_road'] = lerp(0.4, P_mix['flooded_road']**.5, pd.read_csv(BASELINE_LSTM, names=BASELINE_CATEGORIES, index_col=0).sort_index()['flooded_road']**.5)**2
    P_mix = P_mix.div(P_mix.sum(1), 0)
    return P_mix

def submission():
    """
    Returns a single prediction for each object.
    """
    return create_submission(P_ensemble())

if __name__ == '__main__':
    import sys
    output_file, = sys.argv[1:]
    submission().to_csv(output_file)
