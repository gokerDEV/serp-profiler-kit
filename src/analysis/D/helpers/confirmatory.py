import sys
import numpy as np
from src.analysis.D.helpers.stat_utils import run_panel_ols, run_logit_model, check_and_mitigate_collinearity, standardize_variables, apply_winsorization
from src.analysis.D.helpers.report_utils import extract_coeffs

def run_confirmatory_models(current_df, subset_name, dataset_meta, top_k_thresholds,
                            BLOCK_SEMANTICS, BLOCK_READABILITY, BLOCK_PERFORMANCE, BLOCK_ACCESSIBILITY,
                            FDR_WHITELIST):
    results = []
    model_eligibility = []
    model_status = {}
    
    # Base Control
    f_base = "recip_rank ~ C(search_engine, Treatment(reference='google')) + EntityEffects + 1"
    res_base = run_panel_ols(current_df, f_base, f"Base_Control_{subset_name}", [])
    base_r2 = res_base.rsquared if res_base else 0

    # RQ2: Semantics
    m_id = f"RQ2_Semantics_FE_{subset_name}"
    current_semantics, _, _ = check_and_mitigate_collinearity(current_df, BLOCK_SEMANTICS, keep_always=['sim_content'])
    f_fe = f"recip_rank ~ {' + '.join(current_semantics)} + C(search_engine, Treatment(reference='google')) + EntityEffects + 1"
    res_fe = run_panel_ols(current_df, f_fe, m_id, model_eligibility)
    
    rq2_r2 = 0
    if res_fe:
        rq2_r2 = res_fe.rsquared
        delta_r2 = rq2_r2 - base_r2
        rows = extract_coeffs(res_fe, m_id, "FE-continuous", dataset_meta, FDR_WHITELIST)
        for r in rows: r.update({'subset': subset_name, 'delta_r2': delta_r2})
        results.extend(rows)
        model_status[m_id] = "success"
    else:
        model_status[m_id] = "failed"

    # RQ3: Readability
    m_id = f"RQ3_Readability_FE_{subset_name}"
    rq3_preds = current_semantics + BLOCK_READABILITY
    current_rq3_preds, _, _ = check_and_mitigate_collinearity(current_df, rq3_preds, keep_always=current_semantics)
    f_r3 = f"recip_rank ~ {' + '.join(current_rq3_preds)} + C(search_engine, Treatment(reference='google')) + EntityEffects + 1"
    res_r3 = run_panel_ols(current_df, f_r3, m_id, model_eligibility)
    
    if res_r3:
        delta_r2 = res_r3.rsquared - rq2_r2
        rows = extract_coeffs(res_r3, m_id, "FE-continuous", dataset_meta, FDR_WHITELIST)
        for r in rows: r.update({'subset': subset_name, 'delta_r2': delta_r2, 'variant': 'raw'})
        results.extend(rows)
        model_status[m_id] = "success"
        
        # Winsorized
        df_wins = apply_winsorization(current_df, BLOCK_READABILITY)
        m_id_wins = f"RQ3_Readability_FE_Winsorized_{subset_name}"
        res_r3_wins = run_panel_ols(df_wins, f_r3, m_id_wins, [])
        if res_r3_wins:
            rows_w = extract_coeffs(res_r3_wins, m_id_wins, "FE-continuous", dataset_meta, FDR_WHITELIST)
            for r in rows_w: r.update({'subset': subset_name, 'variant': 'winsorized'})
            results.extend(rows_w)

        # Missing data augmented
        loss_pct = (len(current_df) - res_r3.nobs) / len(current_df) * 100
        if loss_pct > 15:
            df_aug = current_df.copy()
            aug_predictors = []
            for col in rq3_preds:
                if df_aug[col].isna().sum() > 0:
                    is_missing_col = f"{col}_is_missing"
                    df_aug[is_missing_col] = df_aug[col].isna().astype(int)
                    df_aug[col] = df_aug[col].fillna(df_aug[col].median())
                    aug_predictors.append(is_missing_col)
            f_aug = f"recip_rank ~ {' + '.join(current_rq3_preds + aug_predictors)} + C(search_engine, Treatment(reference='google')) + EntityEffects + 1"
            m_id_aug = f"RQ3_Readability_FE_IndicatorAugmented_{subset_name}"
            res_aug = run_panel_ols(df_aug, f_aug, m_id_aug, [])
            if res_aug:
                rows_a = extract_coeffs(res_aug, m_id_aug, "FE-continuous", dataset_meta, FDR_WHITELIST)
                for r in rows_a: r.update({'subset': subset_name, 'missing_strategy': 'indicator_augmented'})
                results.extend(rows_a)
    else:
        model_status[m_id] = "failed"

    # Engine Stratified (Full subset only)
    if subset_name == 'Full':
        for engine in current_df['search_engine'].unique():
            m_id_eng = f"RQ3_Readability_FE_{engine}"
            df_eng = current_df[current_df['search_engine'] == engine].copy()
            f_r3_eng = f"recip_rank ~ {' + '.join(BLOCK_SEMANTICS + BLOCK_READABILITY)} + EntityEffects + 1"
            res_eng = run_panel_ols(df_eng, f_r3_eng, m_id_eng, [])
            if res_eng:
                rows_e = extract_coeffs(res_eng, m_id_eng, "FE-continuous", dataset_meta, FDR_WHITELIST)
                for r in rows_e: r.update({'subset': f"Engine_{engine}"})
                results.extend(rows_e)

    # RQ4: Performance
    m_id = f"RQ4_Performance_FE_Std_Beta_{subset_name}"
    cols_to_std_perf = BLOCK_SEMANTICS + BLOCK_PERFORMANCE
    df_std_perf = standardize_variables(current_df, cols_to_std_perf)
    rq4_preds = current_semantics + BLOCK_PERFORMANCE
    current_rq4_preds, _, _ = check_and_mitigate_collinearity(df_std_perf, rq4_preds, keep_always=current_semantics)
    f_r4 = f"recip_rank ~ {' + '.join(current_rq4_preds)} + C(search_engine, Treatment(reference='google')) + EntityEffects + 1"
    res_r4 = run_panel_ols(df_std_perf, f_r4, m_id, model_eligibility)
    
    if res_r4:
        delta_r2 = res_r4.rsquared - rq2_r2
        rows = extract_coeffs(res_r4, m_id, "FE-continuous", dataset_meta, FDR_WHITELIST)
        for r in rows: r.update({'subset': subset_name, 'delta_r2': delta_r2, 'is_standardized': True})
        results.extend(rows)
        model_status[m_id] = "success"
    else:
        model_status[m_id] = "failed"

    # RQ5: Accessibility
    m_id = f"RQ5_Accessibility_FE_Std_Beta_{subset_name}"
    cols_to_std_acc = BLOCK_SEMANTICS + BLOCK_ACCESSIBILITY
    df_std_acc = standardize_variables(current_df, cols_to_std_acc)
    rq5_preds = current_semantics + BLOCK_ACCESSIBILITY
    current_rq5_preds, _, _ = check_and_mitigate_collinearity(df_std_acc, rq5_preds, keep_always=current_semantics)
    f_r5 = f"recip_rank ~ {' + '.join(current_rq5_preds)} + C(search_engine, Treatment(reference='google')) + EntityEffects + 1"
    res_r5 = run_panel_ols(df_std_acc, f_r5, m_id, model_eligibility)

    if res_r5:
        delta_r2 = res_r5.rsquared - rq2_r2
        rows = extract_coeffs(res_r5, m_id, "FE-continuous", dataset_meta, FDR_WHITELIST)
        for r in rows: r.update({'subset': subset_name, 'delta_r2': delta_r2, 'is_standardized': True})
        results.extend(rows)
        model_status[m_id] = "success"
    else:
        model_status[m_id] = "failed"

    # Logit Top-K
    all_metric_cols = BLOCK_SEMANTICS + BLOCK_READABILITY + BLOCK_PERFORMANCE + BLOCK_ACCESSIBILITY
    df_logit_std = standardize_variables(current_df, all_metric_cols)

    for k_val in top_k_thresholds:
        target = f"is_top{k_val}"
        if target not in df_logit_std.columns:
            df_logit_std[target] = (df_logit_std['rank'] <= k_val).astype(int)
        
        # RQ2
        m_id = f"RQ2_Semantics_Logit_Top{k_val}_{subset_name}"
        logit_semantics, _, _ = check_and_mitigate_collinearity(df_logit_std, BLOCK_SEMANTICS, keep_always=['sim_content'])
        f_l2 = f"{target} ~ {' + '.join(logit_semantics)} + C(search_engine, Treatment(reference='google'))"
        res_l2 = run_logit_model(df_logit_std, f_l2, m_id, model_eligibility)
        if res_l2:
             rows = extract_coeffs(res_l2, m_id, f"Top-{k_val} Logit", dataset_meta)
             for r in rows: r['subset'] = subset_name
             results.extend(rows)
             model_status[m_id] = "success"
        else:
             model_status[m_id] = "failed"
        
        # RQ3
        m_id = f"RQ3_Readability_Logit_Top{k_val}_{subset_name}"
        logit_rq3_preds, _, _ = check_and_mitigate_collinearity(df_logit_std, current_semantics + BLOCK_READABILITY, keep_always=logit_semantics)
        f_l3 = f"{target} ~ {' + '.join(logit_rq3_preds)} + C(search_engine, Treatment(reference='google'))"
        res_l3 = run_logit_model(df_logit_std, f_l3, m_id, model_eligibility)
        if res_l3:
             rows = extract_coeffs(res_l3, m_id, f"Top-{k_val} Logit", dataset_meta)
             for r in rows: r['subset'] = subset_name
             results.extend(rows)
             model_status[m_id] = "success"
        else:
             model_status[m_id] = "failed"

        # RQ4
        m_id = f"RQ4_Performance_Logit_Top{k_val}_{subset_name}"
        logit_rq4_preds, _, _ = check_and_mitigate_collinearity(df_logit_std, current_semantics + BLOCK_PERFORMANCE, keep_always=logit_semantics)
        f_l4 = f"{target} ~ {' + '.join(logit_rq4_preds)} + C(search_engine, Treatment(reference='google'))"
        res_l4 = run_logit_model(df_logit_std, f_l4, m_id, model_eligibility)
        if res_l4:
             rows = extract_coeffs(res_l4, m_id, f"Top-{k_val} Logit", dataset_meta)
             for r in rows: r['subset'] = subset_name
             results.extend(rows)
             model_status[m_id] = "success"
        else:
             model_status[m_id] = "failed"

        # RQ5
        m_id = f"RQ5_Accessibility_Logit_Top{k_val}_{subset_name}"
        logit_rq5_preds, _, _ = check_and_mitigate_collinearity(df_logit_std, current_semantics + BLOCK_ACCESSIBILITY, keep_always=logit_semantics)
        f_l5 = f"{target} ~ {' + '.join(logit_rq5_preds)} + C(search_engine, Treatment(reference='google'))"
        res_l5 = run_logit_model(df_logit_std, f_l5, m_id, model_eligibility)
        if res_l5:
             rows = extract_coeffs(res_l5, m_id, f"Top-{k_val} Logit", dataset_meta)
             for r in rows: r['subset'] = subset_name
             results.extend(rows)
             model_status[m_id] = "success"
        else:
             model_status[m_id] = "failed"

    # --- Sequential Interval Model ---
    from src.helpers.data_loader import get_rank_tiers
    tiers_info = get_rank_tiers()
    split_point = tiers_info['logit_cut_points'][-1]
    bins = tiers_info['bins']
    max_rank = bins[bins.index(split_point)+1] if split_point in bins and bins.index(split_point)+1 < len(bins) else 20
    
    df_interval = df_logit_std[df_logit_std['rank'] <= max_rank].copy()
    target_int = f"is_top{split_point}"
    if target_int not in df_interval.columns:
        df_interval[target_int] = (df_interval['rank'] <= split_point).astype(int)
    
    interval_code = f"1_{split_point}_vs_{split_point+1}_{max_rank}"
    interval_label = f"1-{split_point}_vs_{split_point+1}-{max_rank}"
    
    if len(df_interval) > 50 and df_interval[target_int].nunique() > 1:
         # RQ2
         m_id = f"RQ2_Semantics_Interval_{interval_code}_{subset_name}"
         interval_semantics, _, _ = check_and_mitigate_collinearity(df_interval, BLOCK_SEMANTICS, keep_always=['sim_content'])
         f_i2 = f"{target_int} ~ {' + '.join(interval_semantics)} + C(search_engine, Treatment(reference='google'))"
         res_i2 = run_logit_model(df_interval, f_i2, m_id, model_eligibility)
         if res_i2:
             rows = extract_coeffs(res_i2, m_id, "Interval Logit", dataset_meta, FDR_WHITELIST)
             for r in rows: r.update({'subset': f"{subset_name}_Interval", 'interval': interval_label})
             results.extend(rows)
             model_status[m_id] = "success"
         else:
             model_status[m_id] = "failed"

         # RQ3
         m_id = f"RQ3_Readability_Interval_{interval_code}_{subset_name}"
         interval_rq3_preds, _, _ = check_and_mitigate_collinearity(df_interval, BLOCK_SEMANTICS + BLOCK_READABILITY, keep_always=interval_semantics)
         f_i3 = f"{target_int} ~ {' + '.join(interval_rq3_preds)} + C(search_engine, Treatment(reference='google'))"
         res_i3 = run_logit_model(df_interval, f_i3, m_id, model_eligibility)
         if res_i3:
             rows = extract_coeffs(res_i3, m_id, "Interval Logit", dataset_meta, FDR_WHITELIST)
             for r in rows: r.update({'subset': f"{subset_name}_Interval", 'interval': interval_label})
             results.extend(rows)
             model_status[m_id] = "success"
         else:
             model_status[m_id] = "failed"

         # RQ4
         m_id = f"RQ4_Performance_Interval_{interval_code}_{subset_name}"
         interval_rq4_preds, _, _ = check_and_mitigate_collinearity(df_interval, BLOCK_SEMANTICS + BLOCK_PERFORMANCE, keep_always=interval_semantics)
         f_i4 = f"{target_int} ~ {' + '.join(interval_rq4_preds)} + C(search_engine, Treatment(reference='google'))"
         res_i4 = run_logit_model(df_interval, f_i4, m_id, model_eligibility)
         if res_i4:
             rows = extract_coeffs(res_i4, m_id, "Interval Logit", dataset_meta, FDR_WHITELIST)
             for r in rows: r.update({'subset': f"{subset_name}_Interval", 'interval': interval_label})
             results.extend(rows)
             model_status[m_id] = "success"
         else:
             model_status[m_id] = "failed"

         # RQ5
         m_id = f"RQ5_Accessibility_Interval_{interval_code}_{subset_name}"
         interval_rq5_preds, _, _ = check_and_mitigate_collinearity(df_interval, BLOCK_SEMANTICS + BLOCK_ACCESSIBILITY, keep_always=interval_semantics)
         f_i5 = f"{target_int} ~ {' + '.join(interval_rq5_preds)} + C(search_engine, Treatment(reference='google'))"
         res_i5 = run_logit_model(df_interval, f_i5, m_id, model_eligibility)
         if res_i5:
             rows = extract_coeffs(res_i5, m_id, "Interval Logit", dataset_meta, FDR_WHITELIST)
             for r in rows: r.update({'subset': f"{subset_name}_Interval", 'interval': interval_label})
             results.extend(rows)
             model_status[m_id] = "success"
         else:
             model_status[m_id] = "failed"

    return results, model_eligibility, model_status
