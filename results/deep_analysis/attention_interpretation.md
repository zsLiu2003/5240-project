# C. Attention Visualization

## Motivation

Attention rollout is used as qualitative evidence for how the final model processes selected success and failure cases. It should not be interpreted as a causal explanation, but it can show whether chemically important tokens such as charges, metals, halogens, oxygen, boron, or nitrogen receive high attention.

## Figures

| case_id | case_type | figure |
| --- | --- | --- |
| all_selected_cases | combined overview | /hdd2/zesen/daily/5240/5240-project/results/deep_analysis/attention_figures/attention_overview_all_cases.png |
| cid_23718351_1 | largest-error failure | /hdd2/zesen/daily/5240/5240-project/results/deep_analysis/attention_figures/cid_23718351_1.png |
| cid_140455213_2 | halogen failure | /hdd2/zesen/daily/5240/5240-project/results/deep_analysis/attention_figures/cid_140455213_2.png |
| cid_159330689_3 | additional high-error failure | /hdd2/zesen/daily/5240/5240-project/results/deep_analysis/attention_figures/cid_159330689_3.png |
| cid_58676901_4 | additional high-error failure | /hdd2/zesen/daily/5240/5240-project/results/deep_analysis/attention_figures/cid_58676901_4.png |
| cid_22960788_5 | additional high-error failure | /hdd2/zesen/daily/5240/5240-project/results/deep_analysis/attention_figures/cid_22960788_5.png |
| cid_58627710_6 | additional high-error failure | /hdd2/zesen/daily/5240/5240-project/results/deep_analysis/attention_figures/cid_58627710_6.png |
| cid_58604523_7 | additional high-error failure | /hdd2/zesen/daily/5240/5240-project/results/deep_analysis/attention_figures/cid_58604523_7.png |
| cid_129740401_8 | low-error success contrast | /hdd2/zesen/daily/5240/5240-project/results/deep_analysis/attention_figures/cid_129740401_8.png |

## Top-attended Tokens

| case_id | case_type | variant | SMILES | rank | token | attention_rollout |
| --- | --- | --- | --- | --- | --- | --- |
| cid_23718351_1 | largest-error failure | original | [B-]OC.[Na+] | 1 | OC | 0.1220 |
| cid_23718351_1 | largest-error failure | original | [B-]OC.[Na+] | 2 | . | 0.1217 |
| cid_23718351_1 | largest-error failure | original | [B-]OC.[Na+] | 3 | B | 0.1185 |
| cid_23718351_1 | largest-error failure | original | [B-]OC.[Na+] | 4 | -] | 0.0966 |
| cid_23718351_1 | largest-error failure | original | [B-]OC.[Na+] | 5 | [ | 0.0908 |
| cid_140455213_2 | halogen failure | original | [Li+].[O-]I | 1 | +] | 0.1189 |
| cid_140455213_2 | halogen failure | original | [Li+].[O-]I | 2 | . | 0.1147 |
| cid_140455213_2 | halogen failure | original | [Li+].[O-]I | 3 | L | 0.1106 |
| cid_140455213_2 | halogen failure | original | [Li+].[O-]I | 4 | -] | 0.0989 |
| cid_140455213_2 | halogen failure | original | [Li+].[O-]I | 5 | [ | 0.0845 |
| cid_140455213_2 | halogen failure | randomized_1 | I[O-].[Li+] | 1 | -] | 0.1217 |
| cid_140455213_2 | halogen failure | randomized_1 | I[O-].[Li+] | 2 | . | 0.1133 |
| cid_140455213_2 | halogen failure | randomized_1 | I[O-].[Li+] | 3 | L | 0.0988 |
| cid_140455213_2 | halogen failure | randomized_1 | I[O-].[Li+] | 4 | [ | 0.0935 |
| cid_140455213_2 | halogen failure | randomized_1 | I[O-].[Li+] | 5 | i | 0.0916 |
| cid_159330689_3 | additional high-error failure | original | [BH3-]I.[Na+] | 1 | -] | 0.1031 |
| cid_159330689_3 | additional high-error failure | original | [BH3-]I.[Na+] | 2 | I | 0.0993 |
| cid_159330689_3 | additional high-error failure | original | [BH3-]I.[Na+] | 3 | . | 0.0945 |
| cid_159330689_3 | additional high-error failure | original | [BH3-]I.[Na+] | 4 | 3 | 0.0878 |
| cid_159330689_3 | additional high-error failure | original | [BH3-]I.[Na+] | 5 | B | 0.0821 |
| cid_159330689_3 | additional high-error failure | randomized_1 | I[BH3-].[Na+] | 1 | -] | 0.1139 |
| cid_159330689_3 | additional high-error failure | randomized_1 | I[BH3-].[Na+] | 2 | . | 0.0930 |
| cid_159330689_3 | additional high-error failure | randomized_1 | I[BH3-].[Na+] | 3 | H | 0.0903 |
| cid_159330689_3 | additional high-error failure | randomized_1 | I[BH3-].[Na+] | 4 | 3 | 0.0831 |
| cid_159330689_3 | additional high-error failure | randomized_1 | I[BH3-].[Na+] | 5 | [ | 0.0743 |
| cid_58676901_4 | additional high-error failure | original | [B][C-]=N.[Na+] | 1 | = | 0.1211 |
| cid_58676901_4 | additional high-error failure | original | [B][C-]=N.[Na+] | 2 | -] | 0.0934 |
| cid_58676901_4 | additional high-error failure | original | [B][C-]=N.[Na+] | 3 | B | 0.0811 |
| cid_58676901_4 | additional high-error failure | original | [B][C-]=N.[Na+] | 4 | N | 0.0801 |
| cid_58676901_4 | additional high-error failure | original | [B][C-]=N.[Na+] | 5 | . | 0.0754 |
| cid_58676901_4 | additional high-error failure | randomized_1 | [C-](=N)[B].[Na+] | 1 | )[ | 0.1122 |
| cid_58676901_4 | additional high-error failure | randomized_1 | [C-](=N)[B].[Na+] | 2 | ] | 0.0886 |
| cid_58676901_4 | additional high-error failure | randomized_1 | [C-](=N)[B].[Na+] | 3 | (= | 0.0848 |
| cid_58676901_4 | additional high-error failure | randomized_1 | [C-](=N)[B].[Na+] | 4 | . | 0.0771 |
| cid_58676901_4 | additional high-error failure | randomized_1 | [C-](=N)[B].[Na+] | 5 | N | 0.0695 |
| cid_22960788_5 | additional high-error failure | original | CO[O-].[Na+] | 1 | -] | 0.1301 |
| cid_22960788_5 | additional high-error failure | original | CO[O-].[Na+] | 2 | . | 0.1109 |
| cid_22960788_5 | additional high-error failure | original | CO[O-].[Na+] | 3 | [ | 0.0960 |
| cid_22960788_5 | additional high-error failure | original | CO[O-].[Na+] | 4 | N | 0.0937 |
| cid_22960788_5 | additional high-error failure | original | CO[O-].[Na+] | 5 | [ | 0.0905 |
| cid_22960788_5 | additional high-error failure | randomized_1 | O(C)[O-].[Na+] | 1 | -] | 0.1057 |
| cid_22960788_5 | additional high-error failure | randomized_1 | O(C)[O-].[Na+] | 2 | )[ | 0.1040 |
| cid_22960788_5 | additional high-error failure | randomized_1 | O(C)[O-].[Na+] | 3 | . | 0.0835 |
| cid_22960788_5 | additional high-error failure | randomized_1 | O(C)[O-].[Na+] | 4 | ( | 0.0820 |
| cid_22960788_5 | additional high-error failure | randomized_1 | O(C)[O-].[Na+] | 5 | a | 0.0777 |
| cid_58627710_6 | additional high-error failure | original | [B]=[N-].[K+] | 1 | =[ | 0.1117 |
| cid_58627710_6 | additional high-error failure | original | [B]=[N-].[K+] | 2 | -] | 0.1086 |
| cid_58627710_6 | additional high-error failure | original | [B]=[N-].[K+] | 3 | B | 0.0970 |
| cid_58627710_6 | additional high-error failure | original | [B]=[N-].[K+] | 4 | N | 0.0865 |
| cid_58627710_6 | additional high-error failure | original | [B]=[N-].[K+] | 5 | +] | 0.0843 |
| cid_58627710_6 | additional high-error failure | randomized_1 | [K+].[N-]=[B] | 1 | N | 0.1121 |
| cid_58627710_6 | additional high-error failure | randomized_1 | [K+].[N-]=[B] | 2 | . | 0.1098 |
| cid_58627710_6 | additional high-error failure | randomized_1 | [K+].[N-]=[B] | 3 | B | 0.1083 |
| cid_58627710_6 | additional high-error failure | randomized_1 | [K+].[N-]=[B] | 4 | [ | 0.1035 |
| cid_58627710_6 | additional high-error failure | randomized_1 | [K+].[N-]=[B] | 5 | -]=[ | 0.1024 |
| cid_58604523_7 | additional high-error failure | original | C[CH-]N.[Rb+] | 1 | -] | 0.1085 |
| cid_58604523_7 | additional high-error failure | original | C[CH-]N.[Rb+] | 2 | . | 0.1065 |
| cid_58604523_7 | additional high-error failure | original | C[CH-]N.[Rb+] | 3 | +] | 0.0933 |
| cid_58604523_7 | additional high-error failure | original | C[CH-]N.[Rb+] | 4 | N | 0.0932 |
| cid_58604523_7 | additional high-error failure | original | C[CH-]N.[Rb+] | 5 | [ | 0.0859 |
| cid_129740401_8 | low-error success contrast | original | Cl[I-]Cl.[K+] | 1 | -] | 0.1263 |
| cid_129740401_8 | low-error success contrast | original | Cl[I-]Cl.[K+] | 2 | . | 0.1130 |
| cid_129740401_8 | low-error success contrast | original | Cl[I-]Cl.[K+] | 3 | Cl | 0.1113 |
| cid_129740401_8 | low-error success contrast | original | Cl[I-]Cl.[K+] | 4 | [ | 0.0949 |
| cid_129740401_8 | low-error success contrast | original | Cl[I-]Cl.[K+] | 5 | Cl | 0.0896 |
