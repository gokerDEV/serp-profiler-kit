import pandas as pd
import numpy as np

def extract_coeffs(res, model_id, model_family, dataset_meta, whitelist=None,
                   evidence_tier='confirmatory', analysis_tier='confirmatory',
                   anchor='sim_content', model_purpose='confirmatory'):
    from src.helpers.data_loader import get_feature_categories

    if res is None: return []
    if whitelist is None: whitelist = []
    
    FEATURE_CATEGORIES = get_feature_categories()
    rows = []
    
    # Handle API differences
    if hasattr(res, 'params'): # Statsmodels or Linearmodels
        params = res.params
        pvalues = res.pvalues
        
        # CI handling
        if hasattr(res, 'conf_int'):
            conf = res.conf_int()
            # Statsmodels returns numeric cols 0,1. Linearmodels 'lower','upper'.
            if 0 in conf.columns:
                ci_low = conf[0]
                ci_high = conf[1]
            else:
                ci_low = conf['lower']
                ci_high = conf['upper']
        else:
            ci_low = pd.Series(np.nan, index=params.index)
            ci_high = pd.Series(np.nan, index=params.index)
            
        # N and R2
        nobs = getattr(res, 'nobs', getattr(res, 'n', np.nan))
        r2 = getattr(res, 'rsquared', getattr(res, 'prsquared', np.nan)) # prsquared for Logit
    
    for term in params.index:
        # FDR Whitelist & Term Classification
        is_whitelist = term in whitelist
        is_engine_term = term.startswith("C(search_engine")
        is_intercept = term == "Intercept"
        
        # Practical Significance
        practical_flag = "negligible"
        effect_size_val = params[term]
        
        if model_family.startswith("Top"): # Logit -> OR
             or_val = np.exp(effect_size_val)
             if or_val < 0.95 or or_val > 1.05:
                 practical_flag = "practically_significant"
        else: # Continuous -> Beta
             # This function doesn't know if inputs were standardized. 
             # Rely on caller to pass standardized inputs if 'beta*' is desired.
             # If caller passes standardized Dataframe, then coef IS beta*.
             if abs(effect_size_val) >= 0.03: 
                 practical_flag = "potential_significance"

        cat = FEATURE_CATEGORIES.get(term, 'unknown')
        if is_engine_term:
            cat = 'Engine Control'
        elif is_intercept:
            cat = 'Intercept'

        rows.append({
            'model_id': model_id,
            'term': term,
            'category': cat,
            'coef': params[term],
            'pval': pvalues[term],
            'ci_lower': ci_low[term],
            'ci_upper': ci_high[term],
            'n': nobs,
            'r2': r2,
            'model_family': model_family,
            'evidence_tier': evidence_tier,
            'analysis_tier': analysis_tier,
            'anchor': anchor,
            'model_purpose': model_purpose,
            'dataset_id': dataset_meta.get('dataset_id', 'unknown'),
            'dataset_variant': dataset_meta.get('dataset_variant', 'unknown'),
            'code_version': dataset_meta.get('code_version', 'unknown'),
            'generated_at': dataset_meta.get('generated_at', 'unknown'),
            'is_whitelist': is_whitelist,
            'is_engine_term': is_engine_term,
            'is_intercept': is_intercept,
            'practical_flag': practical_flag
        })
        
    return rows
