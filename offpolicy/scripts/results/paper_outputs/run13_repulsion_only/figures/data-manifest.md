# Data Manifest

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| Fig.1 Evacuation-time comparison | D:\github\qmix\offpolicy\scripts\results\paper_outputs\run13_repulsion_only\figures\data\policy_comparison_for_figures.csv | Real | run13 independent 50-episode evaluation | offpolicy/scripts/generate_paper_figures.py | fig1_evacuation_time_comparison.png/svg |
| Fig.2 Cumulative evacuation curves | D:\github\qmix\offpolicy\scripts\results\paper_outputs\run13_repulsion_only\figures\data\cumulative_evacuation_curves.csv | Real | 20 trajectory episodes per policy | offpolicy/scripts/generate_paper_figures.py | fig2_cumulative_evacuation_curve.png/svg |
| Fig.3 Person density heatmaps | person_density_no_robot.csv; person_density_trained_mqmix_best.csv | Real | 20 trajectory episodes per policy | offpolicy/scripts/generate_paper_figures.py | fig3_person_density_heatmaps.png/svg |
| Fig.4 Exit-region density difference | D:\github\qmix\offpolicy\scripts\results\paper_outputs\run13_repulsion_only\figures\data\person_density_no_robot_minus_qmix.csv | Real | 20 trajectory episodes per policy | offpolicy/scripts/generate_paper_figures.py | fig4_exit_density_difference.png/svg |
| Fig.5 Robot position heatmap | D:\github\qmix\offpolicy\scripts\results\paper_outputs\run13_repulsion_only\figures\data\robot_density_trained_mqmix_best.csv | Real | 20 trained-policy trajectory episodes | offpolicy/scripts/generate_paper_figures.py | fig5_robot_position_heatmap.png/svg |
| Fig.6 MQMIX convergence curve | D:\github\qmix\offpolicy\scripts\results\paper_outputs\run13_repulsion_only\figures\data\training_convergence_for_figures.csv | Real | run13 TensorBoard scalar export | offpolicy/scripts/generate_paper_figures.py | fig6_training_convergence.png/svg |
| Fig.7 Repulsion-field scan | D:\github\qmix\offpolicy\scripts\results\paper_outputs\run13_repulsion_only\figures\data\repulsion_scan_top10_for_figures.csv | Real | repulsion_scan_stage2.csv | offpolicy/scripts/generate_paper_figures.py | fig7_repulsion_field_scan.png/svg |
