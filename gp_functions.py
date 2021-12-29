import scipy.stats as st
import numpy as np

def EI_learning(candidates, y_pred, pred_std):
    current_objective = y_pred[np.argmin(y_pred)]
    EI = (current_objective-y_pred)*st.norm.cdf((current_objective-y_pred)/pred_std) \
            +pred_std*st.norm.pdf((current_objective-y_pred)/pred_std)
    new_sample = candidates[np.argmax(EI), :]
    return new_sample, EI
