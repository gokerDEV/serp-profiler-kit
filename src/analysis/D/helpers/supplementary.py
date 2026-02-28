import sys
import numpy as np
from src.analysis.D.helpers.stat_utils import run_panel_ols, run_logit_model, check_and_mitigate_collinearity, standardize_variables
from src.analysis.D.helpers.report_utils import extract_coeffs

def run_supplementary_models(current_df, subset_name, dataset_meta, top_k_thresholds,
                            BLOCK_SEMANTICS, BLOCK_READABILITY, BLOCK_PERFORMANCE, BLOCK_ACCESSIBILITY,
                            FDR_WHITELIST):
    results = []
    model_eligibility = []
    model_status = {}
    
    # Base Control
    f_base = "recip_rank ~ C(search_engine, Treatment(reference='google')) + EntityEffects + 1"
    res_base = run_panel_ols(current_df, f_base, f"Base_Control_{subset_name}", [])
    base_r2 = res_base.rsquared if res_base else 0

    blocks = {
        'Readability': BLOCK_READABILITY,
        'Performance': BLOCK_PERFORMANCE,
        'Accessibility': BLOCK_ACCESSIBILITY
    }

    # Supplementary FE Models
    for block_name, block_cols in blocks.items():
        if not block_cols: continue
        
        m_id = f"Supp_{block_name}_FE_{subset_name}"
        
        # standardize performance/accessibility as in confirmatory models
        if block_name in ['Performance', 'Accessibility']:
            df_model = standardize_variables(current_df, block_cols)
            is_standardized = True
        else:
            df_model = current_df
            is_standardized = False

        # VIF inside block only
        current_preds, _, _ = check_and_mitigate_collinearity(df_model, block_cols, keep_always=[])
        
        if not current_preds: continue
        
        f_fe = f"recip_rank ~ {' + '.join(current_preds)} + C(search_engine, Treatment(reference='google')) + EntityEffects + 1"
        res_fe = run_panel_ols(df_model, f_fe, m_id, model_eligibility)

        if res_fe:
            delta_r2 = res_fe.rsquared - base_r2
            rows = extract_coeffs(res_fe, m_id, "FE-continuous", dataset_meta, FDR_WHITELIST,
                                  evidence_tier='supplementary', analysis_tier='supplementary',
                                  anchor='none', model_purpose='raw_association')
            for r in rows: r.update({'subset': subset_name, 'delta_r2': delta_r2, 'is_standardized': is_standardized})
            results.extend(rows)
            model_status[m_id] = "success"
        else:
            model_status[m_id] = "failed"

    # Supplementary Logit Models
    all_metric_cols = BLOCK_READABILITY + BLOCK_PERFORMANCE + BLOCK_ACCESSIBILITY
    df_logit_std = standardize_variables(current_df, all_metric_cols)

    for k_val in top_k_thresholds:
        target = f"is_top{k_val}"
        if target not in df_logit_std.columns:
            df_logit_std[target] = (df_logit_std['rank'] <= k_val).astype(int)
        
        for block_name, block_cols in blocks.items():
            if not block_cols: continue
            
            m_id = f"Supp_{block_name}_Logit_Top{k_val}_{subset_name}"
            current_preds, _, _ = check_and_mitigate_collinearity(df_logit_std, block_cols, keep_always=[])
            
            if not current_preds: continue
            
            f_l = f"{target} ~ {' + '.join(current_preds)} + C(search_engine, Treatment(reference='google'))"
            res_l = run_logit_model(df_logit_std, f_l, m_id, model_eligibility)
            if res_l:
                 rows = extract_coeffs(res_l, m_id, f"Top-{k_val} Logit", dataset_meta, [],
                                       evidence_tier='supplementary', analysis_tier='supplementary',
                                       anchor='none', model_purpose='raw_association')
                 for r in rows: r['subset'] = subset_name
                 results.extend(rows)
                 model_status[m_id] = "success"
            else:
                 model_status[m_id] = "failed"

    return results, model_eligibility, model_status
